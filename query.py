"""
query.py

Provides a natural language query interface over the enriched knowledge graph
with embeddings in BigQuery. Uses Gemini for query planning.
"""

from google.cloud import bigquery
from vertexai.preview.language_models import TextGenerationModel
from config import (
    PROJECT_ID,
    DATASET_ID,
    DESTINATION_TABLE_ID as TABLE_ID,
    GEN_MODEL_CODER,
    BIGQUERY_LOCATION,
    VERTEX_AI_LOCATION,
)


class EchoDatumEmbeddingAgent:
    def __init__(self, project_id, vertex_location, bq_location, dataset_id, table_id):
        self.project_id = project_id
        self.vertex_location = vertex_location
        self.bq_location = bq_location
        self.dataset_id = dataset_id
        self.table_id = table_id
        self.client = bigquery.Client(project=project_id, location=bq_location)
        self.model = TextGenerationModel.from_pretrained(GEN_MODEL_CODER)

    def _plan_query(self, question: str) -> str:
        """
        Uses Gemini to plan a SQL query against the enriched KG table.
        """
        prompt = f"""
        You are a SQL query planner. The table schema is:

        event_id STRING,
        source_url STRING,
        publish_date DATE,
        subject_entity STRING,
        relationship STRING,
        object_entity STRING,
        latitude FLOAT64,
        longitude FLOAT64,
        country_code STRING,
        net_primary_production FLOAT64,
        dominant_land_cover STRING,
        gbif_species_richness INT64,
        subject_embedding ARRAY<FLOAT64>,
        object_embedding ARRAY<FLOAT64>

        The table name is `{self.project_id}.{self.dataset_id}.{self.table_id}`.

        Write a valid Standard SQL query in BigQuery to answer the following natural language request:
        \"\"\"{question}\"\"\"

        Only output the SQL query. Do not include explanations.
        """
        response = self.model.predict(prompt, temperature=0.0, max_output_tokens=512)
        return response.text.strip()

    def query(self, question: str):
        """
        Executes the planned query and prints results.
        """
        sql = self._plan_query(question)
        print(f"Planned SQL:\n{sql}\n")

        try:
            results = self.client.query(sql).result()
            for row in results:
                print(dict(row))
        except Exception as e:
            print(f"Query failed: {e}")


if __name__ == "__main__":
    agent = EchoDatumEmbeddingAgent(
        project_id=PROJECT_ID,
        vertex_location=VERTEX_AI_LOCATION,
        bq_location=BIGQUERY_LOCATION,
        dataset_id=DATASET_ID,
        table_id=TABLE_ID
    )

    print("\n" + "=" * 80 + "\n")
    # 1. Simple semantic search
    agent.query("Find events about illegal logging or deforestation.")

    print("\n" + "=" * 80 + "\n")
    # 2. Filter by MODIS vegetation productivity (NPP)
    agent.query("List events in areas with very low vegetation productivity (NPP less than 0.2).")

    print("\n" + "=" * 80 + "\n")
    # 3. Aggregate by GBIF species richness
    agent.query("Show the average species richness around reported pollution events.")

    print("\n" + "=" * 80 + "\n")
    # 4. Filter by Dynamic World land cover type
    agent.query("Find events occurring in cropland according to Dynamic World classification.")

    print("\n" + "=" * 80 + "\n")
    # 5. Aggregate counts by country
    agent.query("Count the number of events per country code.")
