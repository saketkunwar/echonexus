"""
enrich_kg_with_embeddings.py

This script enriches the environmental knowledge graph with embeddings
for subject and object entities, using Vertex AI embeddings.
"""

from google.cloud import bigquery
from google.cloud import aiplatform
from config import (
    PROJECT_ID,
    DATASET_ID,
    ENRICHED_KG_TABLE,
    DESTINATION_TABLE_ID,
    EMBEDDING_MODEL_NAME,
    VERTEX_AI_LOCATION,
)

# Initialize Vertex AI
aiplatform.init(project=PROJECT_ID, location=VERTEX_AI_LOCATION)


def _create_destination_table_if_not_exists():
    """
    Creates the destination enriched KG table with embeddings if it does not exist.
    """
    client = bigquery.Client(project=PROJECT_ID)
    table_id = f"{PROJECT_ID}.{DATASET_ID}.{DESTINATION_TABLE_ID}"

    schema = [
        bigquery.SchemaField("event_id", "STRING"),
        bigquery.SchemaField("source_url", "STRING"),
        bigquery.SchemaField("publish_date", "DATE"),
        bigquery.SchemaField("subject_entity", "STRING"),
        bigquery.SchemaField("relationship", "STRING"),
        bigquery.SchemaField("object_entity", "STRING"),
        bigquery.SchemaField("latitude", "FLOAT64"),
        bigquery.SchemaField("longitude", "FLOAT64"),
        bigquery.SchemaField("country_code", "STRING"),

        # Enrichment fields
        bigquery.SchemaField("net_primary_production", "FLOAT64"),
        bigquery.SchemaField("dominant_land_cover", "STRING"),
        bigquery.SchemaField("gbif_species_richness", "INT64"),

        # Embeddings
        bigquery.SchemaField("subject_embedding", "FLOAT", mode="REPEATED"),
        bigquery.SchemaField("object_embedding", "FLOAT", mode="REPEATED"),
    ]

    table = bigquery.Table(table_id, schema=schema)
    table = client.create_table(table, exists_ok=True)
    print(f"Created or verified table {table_id}.")


def _get_events_without_embeddings():
    """
    Get events from the enriched KG table that are missing embeddings.
    """
    client = bigquery.Client(project=PROJECT_ID)
    query = f"""
    SELECT
      event_id,
      source_url,
      publish_date,
      subject_entity,
      relationship,
      object_entity,
      latitude,
      longitude,
      country_code,
      net_primary_production,
      dominant_land_cover,
      gbif_species_richness
    FROM `{PROJECT_ID}.{DATASET_ID}.{ENRICHED_KG_TABLE}`
    WHERE event_id NOT IN (
      SELECT event_id FROM `{PROJECT_ID}.{DATASET_ID}.{DESTINATION_TABLE_ID}`
    )
    """
    return list(client.query(query).result())


def _generate_embedding(text: str):
    """
    Generate embedding for given text using Vertex AI Embeddings.
    """
    if text is None or text.strip() == "":
        return []
    embedding_model = aiplatform.TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)
    embeddings = embedding_model.get_embeddings([text])
    return embeddings[0].values


def enrich_with_embeddings():
    """
    Main function: read KG, add embeddings, write back to BigQuery.
    """
    client = bigquery.Client(project=PROJECT_ID)
    table_id = f"{PROJECT_ID}.{DATASET_ID}.{DESTINATION_TABLE_ID}"

    _create_destination_table_if_not_exists()
    rows = _get_events_without_embeddings()

    enriched_rows = []
    for row in rows:
        subject_emb = _generate_embedding(row.subject_entity)
        object_emb = _generate_embedding(row.object_entity)

        enriched_rows.append({
            "event_id": row.event_id,
            "source_url": row.source_url,
            "publish_date": row.publish_date,
            "subject_entity": row.subject_entity,
            "relationship": row.relationship,
            "object_entity": row.object_entity,
            "latitude": row.latitude,
            "longitude": row.longitude,
            "country_code": row.country_code,
            "net_primary_production": row.net_primary_production,
            "dominant_land_cover": row.dominant_land_cover,
            "gbif_species_richness": row.gbif_species_richness,
            "subject_embedding": subject_emb,
            "object_embedding": object_emb,
        })

    errors = client.insert_rows_json(table_id, enriched_rows)
    if errors:
        print("Encountered errors while inserting rows: ", errors)
    else:
        print(f"Inserted {len(enriched_rows)} rows with embeddings into {table_id}.")


if __name__ == "__main__":
    enrich_with_embeddings()
