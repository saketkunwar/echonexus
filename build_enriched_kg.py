# build_enriched_kg.py

import vertexai
from vertexai.generative_models import GenerativeModel
import pandas as pd
from google.cloud import bigquery
import textwrap
import json
from datetime import datetime, timedelta, timezone

from config import *

GEN_MODEL = GEN_MODEL_BUILD
# ---------------------

class EnrichedKGAgent:

    def __init__(self):
        self.project_id = PROJECT_ID
        self.bq_location = BIGQUERY_LOCATION
        self.vertex_location = VERTEX_AI_LOCATION
        self.enriched_table_id = f"{self.project_id}.{DATASET_ID}.{ENRICHED_KG_TABLE}"

        vertexai.init(project=self.project_id, location=self.vertex_location)
        self.bq_client = bigquery.Client(project=self.project_id)
        self.model = GenerativeModel(GEN_MODEL)
        print("Enriched KG Agent is initialized.")

    def fetch_gdelt_articles(self, region_code: str, themes: list, days_ago: int = 1, limit: int = 10) -> pd.DataFrame:
        """Queries GDELT for recent articles."""
        print(f"\n--- Step 1: Fetching up to {limit} GDELT articles for region '{region_code}' ---")
        theme_conditions = " OR ".join([f"Themes LIKE '%{theme}%'" for theme in themes])
        start_date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
        query = f"""
            SELECT GKGRECORDID as gkg_id, DocumentIdentifier AS url,
                   DATETIME(PARSE_TIMESTAMP("%Y%m%d%H%M%S", CAST(Date AS STRING))) as event_timestamp
            FROM {GDELT_TABLE} WHERE DATE(PARSE_TIMESTAMP("%Y%m%d%H%M%S", CAST(Date AS STRING))) >= "{start_date}"
            AND Locations LIKE '%{region_code}%' AND ({theme_conditions}) ORDER BY Date DESC LIMIT {limit}
        """
        try:
            df = self.bq_client.query(query).to_dataframe()
            print(f"Found {len(df)} articles.")
            return df
        except Exception as e:
            print(f"  ERROR: Could not fetch from GDELT. {e}")
            return pd.DataFrame()

    def extract_base_kg_from_articles(self, articles_df: pd.DataFrame) -> list:
        """Uses an LLM to scrape articles and extract a base KG with geo-coordinates."""
        from newspaper import Article, Config
        print(f"\n--- Step 2: Scraping and extracting base KG from {len(articles_df)} articles ---")
        all_triples = []
        user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'
        config = Config()
        config.browser_user_agent = user_agent
        prompt_template = textwrap.dedent("""
            You are an expert AI for knowledge graph creation. Analyze the article text and extract all meaningful relationships.
            **CRITICAL RULES:**
            1. Extract relationships as a JSON array of objects.
            2. Each object must have these exact keys: "subject_entity", "subject_type", "relationship", "object_entity", "object_type", "location_context", "latitude", "longitude", "country_code".
            3. "latitude" and "longitude" MUST be the approximate geo-coordinates of the "location_context". Return them as numbers, not strings.
            4. "country_code" MUST be the 2-letter ISO code (e.g., "NP" for Nepal).
            5. If a value is not present, use the JSON value `null`.
            6. Normalize entities to lowercase. Omit articles like 'a', 'the'.
            **ARTICLE TEXT:** --- {text} --- **JSON OUTPUT:**
        """)
        for index, row in articles_df.iterrows():
            try:
                article = Article(row['url'], config=config);
                article.download();
                article.parse()
                if not article.text or len(article.text) < 100: continue
                prompt = prompt_template.format(text=article.text[:8000])
                response = self.model.generate_content(prompt)
                cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
                triples = json.loads(cleaned_response)
                for triple in triples:
                    triple['article_id'] = row['gkg_id']
                    triple['article_url'] = row['url']
                    triple['event_timestamp'] = row['event_timestamp']
                all_triples.extend(triples)
                print(f"  Processed article {index + 1}/{len(articles_df)}, extracted {len(triples)} triples.")
            except Exception as e:
                print(f"  Skipped article {row['url']} due to error: {e}")
        return all_triples

    def enrich_kg_with_bq_data(self, kg_triples: list) -> pd.DataFrame:
        """Enriches the extracted KG with data from MODIS NPP and World Bank WDI."""
        if not kg_triples: return pd.DataFrame()

        print(f"\n--- Step 3: Enriching {len(kg_triples)} triples with BigQuery Public Data ---")
        base_df = pd.DataFrame(kg_triples).dropna(subset=['latitude', 'longitude', 'country_code', 'event_timestamp'])
        if base_df.empty:
            print("  No triples with valid geo-coordinates and timestamps to enrich.")
            return pd.DataFrame()

        temp_points_table_id = f"{self.project_id}.{DATASET_ID}._temp_points_to_enrich"
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
        self.bq_client.load_table_from_dataframe(base_df[['latitude', 'longitude', 'country_code', 'event_timestamp']],
                                                 temp_points_table_id, job_config=job_config).result()
        print(f"  Created temporary table with {len(base_df)} points for enrichment.")

        country_codes_for_query = tuple(base_df['country_code'].unique())

        # --- THIS IS THE DEFINITIVE ENRICHMENT QUERY BASED ON THE SUCCESSFUL TESTS ---
        enrichment_query = f"""
        WITH
          -- 1. Get MODIS Net Primary Production using the successful geospatial JOIN
          LandHealth AS (
            SELECT
              p.latitude, p.longitude,
              npp_data.npp * 0.0001 AS net_primary_production
            FROM
              `{temp_points_table_id}` AS p
            JOIN
              `bigquery-public-data.modis_terra_net_primary_production.MODIS_MOD17A3HGF` AS npp_data
              ON ST_CONTAINS(npp_data.geography_polygon, ST_GEOGPOINT(p.longitude, p.latitude))
              AND npp_data.year = EXTRACT(YEAR FROM p.event_timestamp)
          ),
          -- 2. Get latest World Bank WDI data using the successful JOIN
          WDI AS (
            SELECT
              country_code,
              MAX(CASE WHEN indicator_code = 'NY.GDP.PCAP.CD' THEN value END) AS gdp_per_capita_latest,
              MAX(CASE WHEN indicator_code = 'EN.ATM.CO2E.PC' THEN value END) AS co2_emissions_latest
            FROM `bigquery-public-data.world_bank_wdi.indicators_data`
            WHERE
              country_code IN (SELECT DISTINCT country_code FROM `{temp_points_table_id}`)
            GROUP BY country_code
          )
        -- Final SELECT to join all working enrichment sources
        SELECT
          p.latitude, p.longitude, p.country_code,
          lh.net_primary_production,
          wdi.gdp_per_capita_latest, wdi.co2_emissions_latest
        FROM `{temp_points_table_id}` p
        LEFT JOIN LandHealth lh ON p.latitude = lh.latitude AND p.longitude = lh.longitude
        LEFT JOIN WDI wdi ON p.country_code = wdi.country_code
        """
        try:
            # This job MUST run in the 'US' location.
            enrichment_df = self.bq_client.query(enrichment_query, location="US").to_dataframe()
            print(f"  Fetched enrichment data for {len(enrichment_df)} points.")
            final_df = pd.merge(base_df, enrichment_df, on=['latitude', 'longitude', 'country_code'], how='left')
            return final_df
        except Exception as e:
            print(f"  ERROR: Could not fetch enrichment data. {e}")
            return base_df

    def load_kg_to_bigquery(self, final_df: pd.DataFrame):
        """Loads the final, enriched DataFrame into BigQuery."""
        if final_df.empty: print("\n--- No data to load. ---"); return
        print(f"\n--- Step 4: Loading {len(final_df)} enriched triples to BigQuery ---")
        final_df['event_timestamp'] = pd.to_datetime(final_df['event_timestamp'])
        final_df['load_timestamp'] = datetime.now(timezone.utc)
        job_config = bigquery.LoadJobConfig(autodetect=True, write_disposition="WRITE_APPEND")
        try:
            job = self.bq_client.load_table_from_dataframe(final_df, self.enriched_table_id, job_config=job_config)
            job.result()
            print(f"✅ Successfully loaded data to {self.enriched_table_id}.")
        except Exception as e:
            print(f"  ERROR: Could not load data to BigQuery. {e}")

    def run_pipeline(self, region_code: str, themes: list, days_ago: int = 1, limit: int = 10):
        """Orchestrates the entire KG creation and enrichment pipeline."""
        articles_df = self.fetch_gdelt_articles(region_code, themes, days_ago, limit)
        if articles_df.empty: return
        base_triples = self.extract_base_kg_from_articles(articles_df)
        if not base_triples: return
        final_df = self.enrich_kg_with_bq_data(base_triples)
        self.load_kg_to_bigquery(final_df)


if __name__ == "__main__":
    agent = EnrichedKGAgent()
    agent.run_pipeline(region_code=TARGET_REGION, themes=TARGET_THEMES, days_ago=DAYS_AGO, limit=LIMIT_NEWS_ARTICLES)
