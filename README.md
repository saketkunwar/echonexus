# 🌍 EchoNexus – A Neuro-Symbolic Environmental Intelligence Agent

EchoNexus is a **neuro-symbolic pipeline built entirely on Google BigQuery and Vertex AI**.  
It transforms vague, unstructured environmental news into a **high-fidelity, queryable knowledge graph**, enriched with:

- **Narrative signals** from [GDELT](https://www.gdeltproject.org/),
- **Ground-truth physical data** from MODIS Net Primary Production (NPP),
- **Socio-economic context** from World Bank Development Indicators,
- **Semantic embeddings** for deep contextual search.

Users can query this enriched knowledge graph in **plain English**.  
Under the hood, Gemini models translate natural language into SQL, run it inside BigQuery, and return results in **auditable, explainable form**.

---

## 📂 Repository Structure

```
.
├── build_enriched_kg.py          # Builds symbolic knowledge graph from GDELT
├── config.py                     # Configuration (project IDs, models, parameters)
├── create_enriched_schema.py     # Creates schema and tables in BigQuery
├── enrich_kg_with_embeddings.py  # Adds Vertex AI embeddings to graph nodes
├── query_kg_with_embeddings.py   # Natural language query interface (Gemini → SQL)
├── LICENSE                       # License file
├── README.md                     # Project documentation
└── requirements.txt              # Python dependencies
```

---

## ⚙️ Setup

### 1. Environment
- Python 3.9+
- Google Cloud SDK installed and authenticated
- Access to BigQuery and Vertex AI APIs
- Billing enabled in your GCP project

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure
Update `config.py` with your project and dataset information:

```python
PROJECT_ID = "your-gcp-project-id"
DATASET_ID = "your_bigquery_dataset"
VERTEX_AI_LOCATION = "us-central1"
BIGQUERY_LOCATION = "US"

ENRICHED_KG_TABLE = "knowledge_graph_enriched"
DESTINATION_TABLE_ID = "knowledge_graph_final_with_embeddings"
GEN_MODEL_BUILD = "gemini-2.5-flash"
GEN_MODEL_CODER = "gemini-2.5-pro"
```

Optional filters (news volume, themes, region) can also be set:
```python
LIMIT_NEWS_ARTICLES = 6
DAYS_AGO = 5
TARGET_REGION = "NP"
TARGET_THEMES = ["ENV_WATER", "ENV_POLLUTION", "DEFORESTATION", "ENV_MINING"]
```

---

## 🚀 Usage

### 1. Create Schema
First, create the destination tables in BigQuery:
```bash
python create_enriched_schema.py
```

### 2. Build Knowledge Graph
Ingest GDELT news → extract entities, relationships, and geospatial grounding → store in BigQuery:
```bash
python build_enriched_kg.py
```

### 3. Enrich with Embeddings
Add semantic embeddings for subject/object entities using Vertex AI:
```bash
python enrich_kg_with_embeddings.py
```

### 4. Query the Graph
Run the natural language query agent:
```bash
python query_kg_with_embeddings.py
```

The `__main__` section of `query_kg_with_embeddings.py` contains **demo queries**, for example:
```text
"Show me pollution events in regions with Net Primary Production below 0.2"
```
Gemini 2.5 Pro translates this into SQL, runs it in BigQuery, and returns results in natural language.

---

## 🧠 Workflow Overview

1. **Symbolic Layer** – Build a Knowledge Graph of events from GDELT (who, what, where, when).  
2. **Ground-Truth Enrichment** – Join MODIS NPP (vegetation productivity) + World Bank WDI (economic context).  
3. **Neural Layer** – Use Vertex AI embeddings (`text-embedding-004`) for semantic similarity.  
4. **Query Agent** – Gemini 2.5 Pro converts natural language → SQL → BigQuery execution → summarized results.

---

## 📈 Future Enhancements

We prepared scripts to integrate additional datasets via Earth Engine → BigQuery, but could not fully incorporate them due to credit limits:

- **Google Dynamic World (GOOGLE/DYNAMICWORLD/V1):** near real-time 10m land cover classification.  
- **NOAA GSOD Daily Weather:** daily weather conditions from the closest station per event.  

Adding these would extend EchoNexus from historical analysis into **real-time, event-driven intelligence**.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙌 Acknowledgments

- [Google Cloud BigQuery Public Datasets](https://cloud.google.com/bigquery/public-data)  
- [GDELT Project](https://www.gdeltproject.org/)  
- [MODIS NPP](https://modis.gsfc.nasa.gov/data/)  
- [World Bank Open Data](https://data.worldbank.org/)  
- [Vertex AI](https://cloud.google.com/vertex-ai)  
- Gemini family of models for natural language understanding and query planning  
