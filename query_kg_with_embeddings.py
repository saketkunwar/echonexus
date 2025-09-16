import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import TextEmbeddingModel
import pandas as pd
from google.cloud import bigquery
import json

from config import *

# This points to your *final table with embeddings*
TABLE_ID = DESTINATION_TABLE_ID

GEN_MODEL = GEN_MODEL_CODER
EMBED_MODEL = EMBEDDING_MODEL_NAME


class EchoDatumEmbeddingAgent:
    """
    Query agent that supports both normal SQL queries AND semantic similarity queries
    over the embeddings in the enriched KG.
    """

    def __init__(self, project_id: str, vertex_location: str, bq_location: str, dataset_id: str, table_id: str):
        self.project_id = project_id
        self.vertex_location = vertex_location
        self.bq_location = bq_location
        self.fully_qualified_table_id = f"`{project_id}.{dataset_id}.{table_id}`"

        vertexai.init(project=project_id, location=vertex_location)
        self.bq_client = bigquery.Client(project=project_id)
        self.gen_model = GenerativeModel(GEN_MODEL)
        self.embed_model = TextEmbeddingModel.from_pretrained(EMBED_MODEL)

        print("KG Query Agent with Embeddings is initialized and ready.")

    def _plan_query(self, natural_language_question: str) -> dict | None:
        """Step 1: Deconstruct question into JSON plan, including embedding intent."""
        prompt = f"""
        You are a query planner AI. Translate a user question into a structured JSON plan.

        **Table Schema (with embeddings):**
        - subject_entity (STRING), subject_embedding (VECTOR)
        - object_entity (STRING), object_embedding (VECTOR)
        - subject_type, object_type, relationship, location_context, country_code
        - net_primary_production (FLOAT64)
        - gdp_per_capita_latest (FLOAT64)
        - co2_emissions_latest (FLOAT64)

        **Valid intents:**
        - "filter_and_display": SQL filters with normal columns
        - "aggregate": aggregation queries
        - "semantic_search": when the user asks in free text (e.g., "find events about illegal logging")

        For semantic_search:
        {{
          "intent": "semantic_search",
          "search_text": "illegal logging",
          "embedding_column": "subject_embedding",   # or "object_embedding"
          "output_columns": ["subject_entity","relationship","object_entity","location_context"]
        }}

        **User Question:**
        "{natural_language_question}"

        **JSON Plan:**
        """
        try:
            response = self.gen_model.generate_content(prompt)
            plan_text = response.text.strip().replace("```json", "").replace("```", "")
            return json.loads(plan_text)
        except Exception as e:
            print(f"--- ERROR: Could not generate query plan. Error: {e}")
            return None

    def _assemble_sql_from_plan(self, plan: dict) -> str | None:
        """Step 2: Build SQL (for non-embedding queries)."""
        intent = plan.get("intent")

        if intent in ["filter_and_display", "aggregate"]:
            where_clauses = []
            if "filter_conditions" in plan:
                for condition in plan["filter_conditions"]:
                    col, op, val = condition['column'], condition['operator'], condition['value']
                    if isinstance(val, str):
                        val_str, col_str = f"'{val}'", f"LOWER({col})"
                    else:
                        val_str, col_str = str(val), col
                    where_clauses.append(f"{col_str} {op} {val_str}")
            where_clause_str = " AND ".join(where_clauses) if where_clauses else "1=1"

            if intent == "filter_and_display":
                cols = ", ".join(plan.get("output_columns", ["*"]))
                return f"SELECT {cols} FROM {self.fully_qualified_table_id} WHERE {where_clause_str} LIMIT 100;"

            if intent == "aggregate":
                agg_func = plan.get("aggregation_function", "COUNT")
                agg_col = plan.get("aggregation_column", "*")
                group_cols = ", ".join(plan.get("group_by_columns", []))
                if not group_cols: return None
                return f"SELECT {group_cols}, {agg_func}(DISTINCT {agg_col}) AS total FROM {self.fully_qualified_table_id} WHERE {where_clause_str} GROUP BY {group_cols} ORDER BY total DESC;"

        return None

    def _execute_bigquery(self, sql_query: str) -> pd.DataFrame | str:
        try:
            return self.bq_client.query(sql_query, location=self.bq_location).to_dataframe()
        except Exception as e:
            return str(e)

    def _semantic_search(self, plan: dict) -> pd.DataFrame:
        """Runs semantic similarity search with the corrected cosine similarity syntax in BigQuery."""
        search_text = plan.get("search_text", "")
        embedding_col = plan.get("embedding_column", "subject_embedding")

        # We must group by a unique row identifier to get a correct score per row.
        # We select all original columns for grouping to ensure uniqueness.
        group_by_cols = [
            "article_id", "article_url", "subject_entity", "subject_type",
            "relationship", "object_entity", "object_type", "location_context",
            "event_timestamp", "load_timestamp", "latitude", "longitude", "country_code",
            "net_primary_production", "gdp_per_capita_latest", "co2_emissions_latest"
        ]

        # We select the columns the user asked for in the final output.
        output_cols_list = plan.get("output_columns", ["*"])
        if "*" in output_cols_list:
            output_cols = ", ".join([f"kg.{col}" for col in group_by_cols])
        else:
            output_cols = ", ".join([f"kg.{col}" for col in output_cols_list])

        # Generate the embedding for the user's query text.
        query_embedding = self.embed_model.get_embeddings([search_text])[0].values

        sql = f"""
        WITH query_vec AS (
          SELECT @query_vec AS vec
        )
        SELECT
          {output_cols},
          SAFE_DIVIDE(
            SUM(t_val * q_val),
            (SQRT(SUM(t_val * t_val)) * SQRT(SUM(q_val * q_val)))
          ) AS similarity
        FROM
          {self.fully_qualified_table_id} AS kg,
          query_vec,
          -- This is the critical fix: provide unique aliases for the offsets.
          UNNEST(kg.{embedding_col}) AS t_val WITH OFFSET AS t_offset,
          UNNEST(query_vec.vec) AS q_val WITH OFFSET AS q_offset
        WHERE
          -- Join the unnested arrays by their unique offset alias.
          t_offset = q_offset
        GROUP BY
          {", ".join([f"kg.{col}" for col in group_by_cols])} -- Group by all original columns
        ORDER BY
          similarity DESC
        LIMIT 20
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ArrayQueryParameter("query_vec", "FLOAT64", query_embedding)]
        )
        return self.bq_client.query(sql, job_config=job_config, location=self.bq_location).to_dataframe()

    def _summarize_results(self, original_question: str, results_df: pd.DataFrame) -> str:
        if results_df.empty:
            return "I found no data in the knowledge graph that could answer your question."
        prompt = f"Question:\n{original_question}\n\nData (CSV):\n{results_df.to_csv(index=False)}\n\nAnswer:"
        return self.gen_model.generate_content(prompt).text.strip()

    def query(self, natural_language_question: str):
        print(f"\n[USER QUESTION]: {natural_language_question}")
        plan = self._plan_query(natural_language_question)
        if not plan:
            print("[FINAL ANSWER]: Could not generate plan.")
            return
        print(f"--- Plan ---\n{json.dumps(plan, indent=2)}\n")

        if plan["intent"] == "semantic_search":
            print("[Step] 🔍 Running semantic vector search...")
            results = self._semantic_search(plan)
        else:
            sql_query = self._assemble_sql_from_plan(plan)
            if not sql_query:
                print("[FINAL ANSWER]: Invalid SQL plan.")
                return
            print(f"--- SQL ---\n{sql_query}\n")
            results = self._execute_bigquery(sql_query)

        if isinstance(results, str):
            print(f"[ERROR] {results}")
            return

        print(f"--- Found {len(results)} rows ---")
        print(results.head(10).to_string())
        answer = self._summarize_results(natural_language_question, results)
        print(f"\n[FINAL ANSWER]:\n{answer}")


if __name__ == "__main__":
    agent = EchoDatumEmbeddingAgent(
        project_id=PROJECT_ID,
        vertex_location=VERTEX_AI_LOCATION,
        bq_location=BIGQUERY_LOCATION,
        dataset_id=DATASET_ID,
        table_id=TABLE_ID
    )
    '''
    print("\n" + "=" * 80 + "\n")
    # --- Simple Questions to Test the Enriched KG ---
    agent.query("What relationships involve deforestation?")
    print("\n" + "=" * 80 + "\n")
    agent.query("List events in areas with very low vegetation health (NPP less than 0.1).")
    print("\n" + "=" * 80 + "\n")
    agent.query("Find events in countries with high CO2 emissions (greater than 10 metric tons per capita).")
    print("\n" + "=" * 80 + "\n")
    agent.query("Count the number of events per country code.")
    print("\n" + "=" * 80 + "\n")
    agent.query("Which organizations are involved in events in Nepal?")
    '''
    agent.query("Find events about water contamination.")
    print("\n" + "=" * 80 + "\n")

    agent.query("Show me reports of pollution in areas with healthy vegetation (NPP greater than 0.5).")
    print("\n" + "=" * 80 + "\n")

    agent.query("What is the average GDP for events reported in Nepal?")
    print("\n" + "=" * 80 + "\n")

    agent.query("Count all events related to drought.")
    print("\n" + "=" * 80 + "\n")

    agent.query("Find events in countries with high CO2 emissions (greater than 10).")
    print("\n" + "=" * 80 + "\n")
