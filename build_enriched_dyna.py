"""
build_enriched_kg.py

Enriched KG pipeline:
1. Fetch GDELT news articles.
2. Extract base knowledge graph triples with Gemini.
3. Enrich triples with MODIS NPP + GBIF.
4. Enrich with Dynamic World land cover (Earth Engine).
5. Load into BigQuery.
"""

import vertexai
from vertexai.generative_models import GenerativeModel
import pandas as pd
from google.cloud import bigquery
import textwrap
import json
from datetime import datetime, timedelta, timezone

# Earth Engine imports
import ee
import time
from typing import List
from config import *

GEN_MODEL = GEN_MODEL_BUILD

# --- CONSTANTS for Earth Engine ---
DW_COLLECTION_ID = 'GOOGLE/DYNAMICWORLD/V1'
DW_CLASS_NAMES = [
    'water', 'trees', 'grass', 'flooded_vegetation', 'crops',
    'shrub_and_scrub', 'built', 'bare', 'snow_and_ice'
]
BUFFER_METERS = 500
REDUCE_SCALE = 10
REDUCE_MAXPIXELS = int(1e7)


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

    # ---------------- GDELT + Base KG ----------------

    def fetch_gdelt_articles(self, region_code: str, themes: list, days_ago: int = 1, limit: int = 10) -> pd.DataFrame:
        print(f"\n--- Step 1: Fetching up to {limit} GDELT articles for region '{region_code}' ---")
        theme_conditions = " OR ".join([f"Themes LIKE '%{theme}%'" for theme in themes])
        start_date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
        query = f"""
            SELECT GKGRECORDID as gkg_id, DocumentIdentifier AS url,
                   DATETIME(PARSE_TIMESTAMP("%Y%m%d%H%M%S", CAST(Date AS STRING))) as event_timestamp
            FROM {GDELT_TABLE}
            WHERE DATE(PARSE_TIMESTAMP("%Y%m%d%H%M%S", CAST(Date AS STRING))) >= "{start_date}"
              AND Locations LIKE '%{region_code}%' AND ({theme_conditions})
            ORDER BY Date DESC LIMIT {limit}
        """
        try:
            df = self.bq_client.query(query).to_dataframe()
            print(f"Found {len(df)} articles.")
            return df
        except Exception as e:
            print(f"  ERROR: Could not fetch from GDELT. {e}")
            return pd.DataFrame()

    def extract_base_kg_from_articles(self, articles_df: pd.DataFrame) -> list:
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
                article = Article(row['url'], config=config)
                article.download()
                article.parse()
                if not article.text or len(article.text) < 100:
                    continue
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

    # ---------------- BigQuery Enrichment ----------------

    def enrich_kg_with_bq_data(self, kg_triples: list) -> pd.DataFrame:
        """Enriches the extracted KG with MODIS NPP and GBIF."""
        if not kg_triples:
            return pd.DataFrame()

        print(f"\n--- Step 3: Enriching {len(kg_triples)} triples with BigQuery Public Data ---")
        base_df = pd.DataFrame(kg_triples).dropna(subset=['latitude', 'longitude', 'country_code', 'event_timestamp'])
        if base_df.empty:
            print("  No triples with valid geo-coordinates and timestamps to enrich.")
            return pd.DataFrame()

        temp_points_table_id = f"{self.project_id}.{DATASET_ID}._temp_points_to_enrich"
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
        self.bq_client.load_table_from_dataframe(
            base_df[['latitude', 'longitude', 'country_code', 'event_timestamp']],
            temp_points_table_id, job_config=job_config
        ).result()
        print(f"  Created temporary table with {len(base_df)} points for enrichment.")

        enrichment_query = f"""
              WITH
        -- 1. Land health: Net Primary Production (annual)
        LandHealth AS (
          SELECT
            p.latitude, p.longitude, p.event_timestamp,
            npp_data.npp * 0.0001 AS net_primary_production
          FROM `{temp_points_table_id}` p
          LEFT JOIN `bigquery-public-data.modis_terra_net_primary_production.MODIS_MOD17A3HGF` npp_data
            ON ST_CONTAINS(npp_data.geography_polygon, ST_GEOGPOINT(p.longitude, p.latitude))
               AND npp_data.year = EXTRACT(YEAR FROM p.event_timestamp)
        ),

        -- 2. GBIF species richness (within 1 km of event)
        GBIF AS (
          SELECT p.latitude, p.longitude, COUNT(DISTINCT occ.species) as gbif_species_richness
          FROM `{temp_points_table_id}` p
          LEFT JOIN `bigquery-public-data.gbif.occurrences` occ
          ON ST_DWITHIN(
              ST_GEOGPOINT(occ.decimallongitude, occ.decimallatitude),
              ST_GEOGPOINT(p.longitude, p.latitude),
              1000
          )
          GROUP BY p.latitude, p.longitude
        )

        SELECT
          p.*,
          lh.net_primary_production,
          gbif.gbif_species_richness
        FROM `{temp_points_table_id}` p
        LEFT JOIN LandHealth lh
          ON p.latitude = lh.latitude
         AND p.longitude = lh.longitude
         AND p.event_timestamp = lh.event_timestamp
        LEFT JOIN GBIF gbif
          ON p.latitude = gbif.latitude
         AND p.longitude = gbif.longitude
        """

        try:
            enrichment_df = self.bq_client.query(enrichment_query, location="US").to_dataframe()
            print(f"  Fetched enrichment data for {len(enrichment_df)} points.")
            final_df = pd.merge(base_df, enrichment_df,
                                on=['latitude', 'longitude', 'country_code'], how='left')
            return final_df
        except Exception as e:
            print(f"  ERROR: Could not fetch enrichment data. {e}")
            return base_df

    # ---------------- Earth Engine Enrichment ----------------

    def enrich_with_dynamic_world_from_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Call Earth Engine Dynamic World to enrich with dominant land cover."""
        if df.empty:
            return df

        print(f"\n--- Step 3b: Enriching {len(df)} points with Dynamic World land cover ---")
        try:
            ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
        except Exception as e:
            print(f"  ERROR: Could not initialize Earth Engine. {e}")
            return df

        # prepare unique points
        points_df = df[['latitude', 'longitude', 'event_timestamp']].drop_duplicates()
        features: List[ee.Feature] = []
        for _, row in points_df.iterrows():
            try:
                ts = pd.Timestamp(row['event_timestamp']).tz_convert("UTC")
            except Exception:
                continue
            geom = ee.Geometry.Point(float(row['longitude']), float(row['latitude']))
            feat = ee.Feature(geom, {
                'latitude': float(row['latitude']),
                'longitude': float(row['longitude']),
                'event_timestamp': int(ts.timestamp() * 1000)
            })
            features.append(feat)

        if not features:
            return df

        fc = ee.FeatureCollection(features)

        def safe_pick_dominant(feat, prob_dict):
            prob_dict = ee.Dictionary(prob_dict)
            keys = prob_dict.keys()
            vals = prob_dict.values()
            safe_vals = ee.List(vals).map(lambda v: ee.Number(v).double().divide(1).orElse(-1))
            max_val = ee.Number(safe_vals.reduce(ee.Reducer.max()))
            idx = safe_vals.indexOf(max_val)
            dominant_key = keys.get(idx)
            return ee.Algorithms.If(
                dominant_key,
                feat.set('dominant_land_cover', dominant_key),
                feat.set('dominant_land_cover', 'no_data')
            )

        def get_dominant(feature):
            pt = feature.geometry()
            dt = ee.Date(feature.get('event_timestamp'))
            buffered = pt.transform('EPSG:3857', 1).buffer(BUFFER_METERS)
            aoib = buffered.bounds(1)
            dw_img = (ee.ImageCollection(DW_COLLECTION_ID)
                      .filterBounds(aoib)
                      .filterDate(dt.advance(-60, 'day'), dt.advance(1, 'day'))
                      .sort('system:time_start', False)
                      .first())
            def process(img):
                mean_probs = img.select(DW_CLASS_NAMES).reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=aoib,
                    scale=REDUCE_SCALE,
                    maxPixels=REDUCE_MAXPIXELS,
                    bestEffort=True
                )
                return ee.Algorithms.If(
                    mean_probs,
                    safe_pick_dominant(feature, mean_probs),
                    feature.set('dominant_land_cover', 'no_data')
                )
            return ee.Algorithms.If(dw_img, process(dw_img), feature.set('dominant_land_cover', 'no_image'))

        mapped = fc.map(get_dominant)
        try:
            results = mapped.getInfo()
        except Exception as e:
            print(f"  ERROR: Earth Engine job failed {e}")
            return df

        rows = []
        for f in results.get('features', []):
            props = f.get('properties', {})
            rows.append({
                'latitude': props.get('latitude'),
                'longitude': props.get('longitude'),
                'dominant_land_cover': props.get('dominant_land_cover')
            })
        ee_df = pd.DataFrame(rows)
        print(f"  Dynamic World enrichment completed for {len(ee_df)} points.")
        return pd.merge(df, ee_df, on=['latitude', 'longitude'], how='left')

    # ---------------- Load to BigQuery ----------------

    def load_kg_to_bigquery(self, final_df: pd.DataFrame):
        if final_df.empty:
            print("\n--- No data to load. ---")
            return
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

    # ---------------- Pipeline ----------------

    def run_pipeline(self, region_code: str, themes: list, days_ago: int = 1, limit: int = 10):
        articles_df = self.fetch_gdelt_articles(region_code, themes, days_ago, limit)
        if articles_df.empty:
            return
        base_triples = self.extract_base_kg_from_articles(articles_df)
        if not base_triples:
            return
        final_df = self.enrich_kg_with_bq_data(base_triples)
        final_df = self.enrich_with_dynamic_world_from_df(final_df)
        self.load_kg_to_bigquery(final_df)


if __name__ == "__main__":
    agent = EnrichedKGAgent()
    agent.run_pipeline(region_code=TARGET_REGION, themes=TARGET_THEMES, days_ago=DAYS_AGO, limit=LIMIT_NEWS_ARTICLES)
