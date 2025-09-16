import vertexai
from vertexai.language_models import TextEmbeddingModel
import pandas as pd
from google.cloud import bigquery
import time

from config import *
SOURCE_TABLE_ID = ENRICHED_KG_TABLE

class EnrichedKGAgent:
    """
    Enriches the knowledge graph:
    Read from BigQuery -> Get embeddings from Vertex AI -> Write back to BigQuery.
    """

    def __init__(self):
        self.project_id = PROJECT_ID
        self.bq_location = BIGQUERY_LOCATION
        self.vertex_location = VERTEX_AI_LOCATION
        self.source_table_ref = f"{self.project_id}.{DATASET_ID}.{SOURCE_TABLE_ID}"
        self.dest_table_ref = f"{self.project_id}.{DATASET_ID}.{DESTINATION_TABLE_ID}"

        vertexai.init(project=self.project_id, location=self.vertex_location)
        self.bq_client = bigquery.Client(project=self.project_id)
        self.embedding_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)
        print("Enriched KG Agent initialized.")

    def _create_destination_table_if_not_exists(self):
        """Creates the destination table with the final schema."""
        schema = [
            bigquery.SchemaField("article_id", "STRING"),
            bigquery.SchemaField("article_url", "STRING"),
            bigquery.SchemaField("subject_entity", "STRING"),
            bigquery.SchemaField("subject_type", "STRING"),
            bigquery.SchemaField("relationship", "STRING"),
            bigquery.SchemaField("object_entity", "STRING"),
            bigquery.SchemaField("object_type", "STRING"),
            bigquery.SchemaField("location_context", "STRING"),
            bigquery.SchemaField("event_timestamp", "TIMESTAMP"),
            bigquery.SchemaField("load_timestamp", "TIMESTAMP"),
            bigquery.SchemaField("latitude", "FLOAT64"),
            bigquery.SchemaField("longitude", "FLOAT64"),
            bigquery.SchemaField("country_code", "STRING"),
            bigquery.SchemaField("net_primary_production", "FLOAT64"),
            bigquery.SchemaField("gdp_per_capita_latest", "FLOAT64"),
            bigquery.SchemaField("co2_emissions_latest", "FLOAT64"),
            bigquery.SchemaField("subject_embedding", "FLOAT", mode="REPEATED"),
            bigquery.SchemaField("object_embedding", "FLOAT", mode="REPEATED"),
        ]
        table = bigquery.Table(self.dest_table_ref, schema=schema)
        self.bq_client.create_table(table, exists_ok=True)
        print(f"Destination table '{self.dest_table_ref}' is ready.")

    def _get_unprocessed_article_ids(self) -> list:
        """Gets article_ids that have not yet been processed."""
        query = f"""
            SELECT t1.article_id
            FROM `{self.source_table_ref}` AS t1
            LEFT JOIN `{self.dest_table_ref}` AS t2
            ON t1.article_id = t2.article_id
            WHERE t2.article_id IS NULL AND t1.article_id IS NOT NULL
        """
        print("Finding unprocessed articles...")
        df = self.bq_client.query(query, location=self.bq_location).to_dataframe()
        return df['article_id'].unique().tolist()

    def _get_embeddings_for_texts(self, texts: list) -> list:
        """Safely gets embeddings in sub-batches of ≤250 texts."""
        all_embeddings = []
        try:
            for i in range(0, len(texts), API_BATCH_SIZE):
                sub_batch = texts[i:i + API_BATCH_SIZE]
                # Skip completely empty batches (should not happen)
                if not sub_batch:
                    continue
                instances = self.embedding_model.get_embeddings(sub_batch)
                all_embeddings.extend([instance.values for instance in instances])
            return all_embeddings
        except Exception as e:
            print(f"  WARNING: Vertex AI API call failed. Error: {e}. Returning empty embeddings.")
            return [None] * len(texts)

    def run_pipeline(self):
        """Main enrichment pipeline."""
        self._create_destination_table_if_not_exists()

        article_ids_to_process = self._get_unprocessed_article_ids()
        if not article_ids_to_process:
            print("All articles are already processed. Nothing to do.")
            return

        print(f"Found {len(article_ids_to_process)} articles to process. Starting in batches of {READ_BATCH_SIZE}...")

        total_processed = 0
        for i in range(0, len(article_ids_to_process), READ_BATCH_SIZE):
            batch_ids = article_ids_to_process[i:i + READ_BATCH_SIZE]
            print(f"\n--- Processing batch {i//READ_BATCH_SIZE + 1} of {(len(article_ids_to_process)-1)//READ_BATCH_SIZE + 1} ---")

            # 1. READ
            batch_ids_sql = ", ".join([f"'{id}'" for id in batch_ids])
            read_query = f"SELECT * FROM `{self.source_table_ref}` WHERE article_id IN ({batch_ids_sql})"
            batch_df = self.bq_client.query(read_query, location=self.bq_location).to_dataframe()

            print(f"  Read {len(batch_df)} rows. Now generating embeddings...")

            # 2. PROCESS
            subject_texts = batch_df['subject_entity'].fillna('').astype(str).tolist()
            object_texts = batch_df['object_entity'].fillna('').astype(str).tolist()

            # Replace empty/whitespace-only strings with a safe placeholder
            subject_texts = [t if t.strip() else "[EMPTY]" for t in subject_texts]
            object_texts = [t if t.strip() else "[EMPTY]" for t in object_texts]

            subject_embeddings = self._get_embeddings_for_texts(subject_texts)
            object_embeddings = self._get_embeddings_for_texts(object_texts)

            batch_df['subject_embedding'] = subject_embeddings
            batch_df['object_embedding'] = object_embeddings

            # Drop rows where embedding generation failed
            batch_df.dropna(subset=['subject_embedding', 'object_embedding'], inplace=True)
            if batch_df.empty:
                print("  No embeddings were successfully generated for this batch. Skipping.")
                continue

            # 3. WRITE
            print(f"  Embeddings generated. Now loading {len(batch_df)} enriched rows to BigQuery...")
            job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
            job = self.bq_client.load_table_from_dataframe(batch_df, self.dest_table_ref, job_config=job_config)
            job.result()
            total_processed += len(batch_df)
            print(f"✅ Batch loaded successfully.")

        print(f"\n--- Final Enrichment Complete! ---")
        print(f"Successfully processed and loaded a total of {total_processed} articles.")


if __name__ == "__main__":
    agent = EnrichedKGAgent()
    agent.run_pipeline()
