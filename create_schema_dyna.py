from google.cloud import bigquery
from config import PROJECT_ID, DATASET_ID, BIGQUERY_LOCATION

def create_dataset_if_not_exists():
    """
    Creates the dataset if it does not exist already.
    """
    client = bigquery.Client(project=PROJECT_ID)
    dataset_id = f"{PROJECT_ID}.{DATASET_ID}"
    dataset = bigquery.Dataset(dataset_id)
    dataset.location = BIGQUERY_LOCATION

    try:
        client.get_dataset(dataset_id)  # Check if dataset exists
        print(f"Dataset {dataset_id} already exists.")
    except Exception:
        client.create_dataset(dataset, exists_ok=True)
        print(f"Created dataset {dataset_id}.")

def create_enriched_schema():
    client = bigquery.Client(project=PROJECT_ID)
    dataset_ref = client.dataset(DATASET_ID)

    table_id = f"{PROJECT_ID}.{DATASET_ID}.knowledge_graph_enriched"

    schema = [
        bigquery.SchemaField("event_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("source_url", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("publish_date", "DATE", mode="NULLABLE"),
        bigquery.SchemaField("subject_entity", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("object_entity", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("relationship", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("latitude", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("longitude", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("country_code", "STRING", mode="NULLABLE"),

        # MODIS NPP enrichment
        bigquery.SchemaField("net_primary_production", "FLOAT64", mode="NULLABLE"),

        # Dynamic World enrichment
        bigquery.SchemaField("dominant_land_cover", "STRING", mode="NULLABLE"),

        # GBIF enrichment
        bigquery.SchemaField("gbif_species_richness", "INT64", mode="NULLABLE"),
    ]

    table = bigquery.Table(table_id, schema=schema)
    table = client.create_table(table, exists_ok=True)
    print(f"Created or updated table {table_id} with schema.")

if __name__ == "__main__":
    create_dataset_if_not_exists()
    create_enriched_schema()
