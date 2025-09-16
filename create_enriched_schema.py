# create_enriched_schema.py

from google.cloud import bigquery
from config import *
TABLE_ID = ENRICHED_KG_TABLE


def create_enriched_knowledge_graph_table():
    """
    Creates the final, enriched KG table, ensuring the clustering specification
    matches the existing table's required configuration.
    """
    client = bigquery.Client(project=PROJECT_ID)
    table_ref = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"

    ddl_statement = f"""
    CREATE OR REPLACE TABLE `{table_ref}`
    (
      -- Core KG Triple Columns
      article_id STRING,
      article_url STRING,
      subject_entity STRING,
      subject_type STRING,
      relationship STRING,
      object_entity STRING,
      object_type STRING,
      location_context STRING,
      event_timestamp TIMESTAMP,
      load_timestamp TIMESTAMP,

      -- Geospatial & Country Context (extracted by LLM)
      latitude FLOAT64,
      longitude FLOAT64,
      country_code STRING,

      -- MODIS Net Primary Production Enrichment
      net_primary_production FLOAT64,

      -- World Bank WDI Enrichment
      gdp_per_capita_latest FLOAT64,
      co2_emissions_latest FLOAT64
    )
    PARTITION BY
      DATE(event_timestamp)
    -- --- THE DEFINITIVE FIX ---
    -- The CLUSTER BY clause is updated to match the existing table's specification.
    CLUSTER BY
      country_code, subject_type, relationship
    -- --- END OF FIX ---
    OPTIONS(
      description="An enriched KG fusing GDELT events with MODIS NPP and World Bank WDI data."
    );
    """
    print(f"Attempting to create or replace enriched table: {table_ref}")
    try:
        query_job = client.query(ddl_statement, location=LOCATION)
        query_job.result()
        print(f"✅ Successfully created or replaced table '{table_ref}'.")
    except Exception as e:
        print(f"An error occurred while creating the table: {e}")


if __name__ == "__main__":
    create_enriched_knowledge_graph_table()
