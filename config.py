# --- CONFIGURATION ---
# PROJECT_ID and DATASET_ID must be set by user t their own projects
PROJECT_ID = "plated-dryad-459812-f3"
DATASET_ID = "agent"

VERTEX_AI_LOCATION = "us-central1"
BIGQUERY_LOCATION = "US"
GDELT_TABLE = "`gdelt-bq.gdeltv2.gkg`"
ENRICHED_KG_TABLE = "knowledge_graph_enriched"
# Use the latest stable embedding model
EMBEDDING_MODEL_NAME = "text-embedding-004"
DESTINATION_TABLE_ID = "knowledge_graph_final_with_embeddings"

GEN_MODEL_BUILD = "gemini-2.5-flash"
GEN_MODEL_CODER = "gemini-2.5-pro"


LIMIT_NEWS_ARTICLES = 6
DAYS_AGO = 5
TARGET_REGION = "NP"
TARGET_THEMES = ["ENV_WATER", "ENV_POLLUTION", "DEFORESTATION", "ENV_MINING"]


# --- Batch Processing Configuration ---
READ_BATCH_SIZE = 1000        # Rows to pull from BigQuery per batch
API_BATCH_SIZE = 250          # Vertex AI Embeddings API limit
# --------------------------------------------------------------------
