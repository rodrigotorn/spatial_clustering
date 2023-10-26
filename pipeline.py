# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import geopandas as gpd
import libpysal as sal

import src.functions as f
from src.logger import get_logger

import logging
logger = get_logger('src', logging.INFO)

# %%
np.random.seed(0)

INITIAL_CLUSTERS_COUNT: int = 200
MIN_RESIDENTS_PER_CLUSTER: int = 30000

# %%
grid: gpd.GeoDataFrame = f.read_sp_geographic_data(
  'data/geographic_data/35SEE250GC_SIR.shp'
)
data: pd.DataFrame = f.read_sp_demographic_data('data/demographic_data/Basico_SP1.csv')
df_by_sector: gpd.GeoDataFrame = grid.merge(data, on='id', how='left')
del grid, data

weights: sal.weights.Queen = sal.weights.Queen.from_dataframe(df_by_sector)
df_by_sector, missing_sectors = f.fill_missing_data(
  df_by_sector,
  weights
)
weights = sal.weights.Queen.from_dataframe(df_by_sector)

df_by_sector = f.agglomerative_cluster(
  df_by_sector,
  INITIAL_CLUSTERS_COUNT,
  weights
)
df_by_sector = f.recluster_small_clusters(
  df_by_sector,
  MIN_RESIDENTS_PER_CLUSTER,
  weights
)
df_by_sector = f.manually_fill_remaining_sectors(
  df_by_sector,
  missing_sectors
)
df_by_region: gpd.GeoDataFrame = \
  df_by_sector.dissolve(by='cluster', as_index=False)

f.plot_regions(df_by_region)
