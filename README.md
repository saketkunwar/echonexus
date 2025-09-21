# EchoNexus: High-Fidelity Neuro-Symbolic Agent for Environmental Intelligence

## Overview
EchoNexus transforms unstructured environmental news into **structured, verifiable intelligence**.  
Using a **Neuro-Symbolic pipeline** built on **Google BigQuery AI** with Gemini models, it fuses:  
- **News text (GDELT)**  
- **Earth observation (MODIS Net Primary Production + Dynamic World land cover)**  
- **Biodiversity indicators (GBIF species richness)**  

The result is a **knowledge graph** enriched with both symbolic structure and neural semantic embeddings, enabling **natural language queries in BigQuery**.

---

## Key Features
- **Knowledge Graph Backbone**  
  - Extracts entities, relationships, and geospatial context from GDELT articles using **Gemini 2.5**.  
  - Events are grounded with **latitude/longitude + ISO country code**.  

- **Ground-Truth Enrichment**  
  - **MODIS NPP** (`bigquery-public-data.modis_terra_net_primary_production.MODIS_MOD17A3HGF`)  
    → Annual vegetation productivity (kg C/m²/year).  
  - **GBIF Species Richness** (`bigquery-public-data.gbif.occurrences`)  
    → Counts distinct species occurrences within 1 km of each event.  
  - **Dynamic World Land Cover** (`GOOGLE/DYNAMICWORLD/V1` via Earth Engine)  
    → Adds dominant land cover type (trees, crops, water, built-up).  

- **Neural Semantic Layer**  
  - Embeds entities using **Vertex AI `text-embedding-004`**.  
  - Supports **semantic similarity search** via BigQuery **VECTOR_SEARCH**.  

- **Natural Language Q&A**  
  - **Gemini 2.5 Pro** converts natural language into SQL queries.  
  - Supports **hybrid querying** (symbolic + semantic).  

---

## Repository Structure
```
.
├── build_enriched_dyna.py        # Pipeline: fetch GDELT, extract KG, enrich with MODIS + GBIF + Dynamic World
├── create_schema_dyna.py         # Creates BigQuery dataset & table schema
├── enrich_kg_dyna_embeddings.py  # Adds embeddings to KG entities using Vertex AI
├── query.py                      # Natural language → SQL querying with Gemini
├── ee_enrichment.py              # Earth Engine integration for Dynamic World enrichment
├── config.py                     # User configuration (project_id, dataset_id, region, etc.)
├── requirements.txt              # Python dependencies
├── LICENSE                       # License file
├── README.md                     # Project documentation
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

   Includes:
   - `google-cloud-bigquery`
   - `google-cloud-aiplatform`
   - `earthengine-api`
   - `pandas`, `newspaper3k`, etc.

3. **Configure environment**  
   Edit `config.py` with:
   - `PROJECT_ID` = your GCP project ID  
   - `DATASET_ID` = target BigQuery dataset  
   - `VERTEX_AI_LOCATION` & `BIGQUERY_LOCATION`  

4. **Authenticate Google Cloud & Earth Engine**
   ```bash
   gcloud auth application-default login
   earthengine authenticate
   ```

5. **Run the pipeline**
   ```bash
   python create_schema_dyna.py
   python build_enriched_dyna.py
   python enrich_kg_dyna_embeddings.py
   python query.py
   ```

---

## Example Queries
- **Semantic Search**
  ```sql
  Find events about illegal logging or deforestation.
  ```

- **Filter by MODIS NPP**
  ```sql
  List events in areas with vegetation productivity (NPP) < 0.2.
  ```

- **Aggregate by GBIF species richness**
  ```sql
  Show the average species richness around reported pollution events.
  ```

- **Land cover context**
  ```sql
  Find events occurring in cropland.
  ```

- **Country aggregation**
  ```sql
  Count the number of events per country.
  ```

---

## Future Improvements
- Integrate **NOAA weather** and **socio-economic indicators** later if needed.  
- Develop **dashboards & APIs** for ESG stakeholders.  

---

## License
MIT License. See `LICENSE` file.
