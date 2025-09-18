# EchoNexus: High-Fidelity Neuro-Symbolic Agent for Environmental Intelligence

## Overview
EchoNexus transforms vague, unstructured environmental news into **structured, verifiable intelligence**.  
Using a **Neuro-Symbolic pipeline** built entirely in **Google BigQuery** with Gemini models, it fuses:
- **News text (GDELT)**
- **Earth observation (MODIS + Dynamic World)**
- **Biodiversity indicators (GBIF species richness)**  

The result is a **queryable knowledge graph** enriched with both symbolic structure and neural semantic understanding, enabling natural language questions to be translated into precise BigQuery queries.

---

## Key Features
- **Knowledge Graph Backbone**  
  - Extract entities, relationships, and geospatial context from GDELT articles using **Gemini 2.5**.  
  - Each event is grounded with latitude/longitude and ISO country code, bridging text with spatial data.  

- **Ground-Truth Enrichment**
  - **MODIS NPP** (`bigquery-public-data.modis_terra_net_primary_production.MODIS_MOD17A3HGF`)  
    Adds annual **net primary production** values to measure vegetation productivity and detect physical impacts of deforestation or pollution.  
  - **GBIF Species Richness** (`bigquery-public-data.gbif.occurrences`)  
    Adds biodiversity context by counting distinct species occurrences within 1 km of each event location.  
  - **Dynamic World Land Cover** (`GOOGLE/DYNAMICWORLD/V1` via Earth Engine)  
    Enriches each event with **dominant land cover type** (e.g., trees, crops, water), giving context on where the event happened.  

- **Neural Semantic Layer**
  - Embeds entities using **BigQuery ML.GENERATE_EMBEDDING** with `text-embedding-004`.  
  - Enables **semantic similarity search** (e.g., “deluge” ≈ “flooding”) instead of brittle keyword filters.  

- **Natural Language Q&A**
  - **Gemini 2.5 Pro** converts user questions into executable SQL queries.  
  - Hybrid planner decides when to use **semantic search** (neural) vs. **symbolic filters** (numeric thresholds).  
  - Returns results as enriched tables or maps with direct grounding in environmental data.

---

## Repository Structure
```
.
├── build_enriched_kg.py          # Pipeline to fetch GDELT, extract KG, enrich with MODIS + GBIF + Dynamic World
├── create_enriched_schema.py     # Creates BigQuery tables for storing KG
├── enrich_kg_with_embeddings.py  # Adds embeddings (ML.GENERATE_EMBEDDING) to KG nodes
├── query_kg_with_embeddings.py   # Demo agent: natural language → SQL queries with semantic + symbolic filters
├── config.py                     # User configuration (project_id, dataset_id, region, etc.)
├── requirements.txt              # Python dependencies
└── LICENSE / README.md
```

---

## Setup Instructions

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-repo/echonexus.git
   cd echonexus
   ```

2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

   Requirements include:
   - `google-cloud-bigquery`
   - `google-cloud-aiplatform`
   - `earthengine-api`
   - `pandas`, `newspaper3k`, etc.

3. **Configure your environment**  
   Edit `config.py` with:
   - `PROJECT_ID` = your GCP project
   - `DATASET_ID` = target BigQuery dataset
   - `VERTEX_AI_LOCATION` and `BIGQUERY_LOCATION` (default: `us-central1`, `US`)

4. **Authenticate Google Cloud & Earth Engine**  
   ```bash
   gcloud auth application-default login
   earthengine authenticate
   ```

5. **Run the pipeline**  
   - Create schema:
     ```bash
     python create_enriched_schema.py
     ```
   - Build enriched KG:
     ```bash
     python build_enriched_kg.py
     ```
   - Add embeddings:
     ```bash
     python enrich_kg_with_embeddings.py
     ```
   - Query with agent:
     ```bash
     python query_kg_with_embeddings.py
     ```

---

## Example Queries
The Q&A agent supports hybrid symbolic + semantic search. Examples:

1. **Semantic search for deforestation events**
   ```
   Find events about illegal logging or deforestation.
   ```

2. **Filter by MODIS vegetation health**
   ```
   List events in areas with very low vegetation productivity (NPP < 0.2).
   ```

3. **Aggregate biodiversity context**
   ```
   Show the average species richness around reported pollution events.
   ```

4. **Filter by Dynamic World land cover**
   ```
   Find events occurring in cropland.
   ```

5. **Country-level aggregation**
   ```
   Count the number of events per country.
   ```

---

## Future Improvements
- Add **real-time Dynamic World feeds** for fresher context.  
- Integrate **daily climate/weather data** (NOAA GSOD) for immediate impact assessments.  
- Expand socio-economic context with **World Bank WDI** when resource limits allow.  

---

## License
MIT License. See `LICENSE` file for details.

---

## Acknowledgments
- Google BigQuery Public Datasets  
- Google Earth Engine Dynamic World  
- GBIF Occurrence Data  
- MODIS Net Primary Production (NASA)  
- Gemini 2.5 Models (Vertex AI)  

---
