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
import test.test_functions as test_f
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
df_by_region = df_by_region[['cluster', 'geometry']]
df_by_region['cluster'] = [x for x in range(1, len(df_by_region['cluster']) + 1)]

f.plot_sectors(df_by_sector)
f.plot_regions(df_by_region)

logger.info('Saving results file into outputs folder')
df_by_region.to_file('outputs/regions.shp', index=False)

# %%
f.plot_example(
  df=test_f.gen_simulated_sectors(4),
  fname='base_example.png'
)

missing_df_step_1 = test_f.gen_simulated_sectors(4)
for i in [2, 3, 6, 7, 13]:
  missing_df_step_1.loc[i, 'value'] = np.nan

f.plot_example(
  df=missing_df_step_1,
  fname='missing_df_step_1.png'
)

missing_df_step_2 = f.fill_with_neighbors_data(
  df=missing_df_step_1,
  agg_dict={'value': 'mean'},
  weights=sal.weights.Queen.from_dataframe(missing_df_step_1)
)

f.plot_example(
  df=missing_df_step_2,
  fname='missing_df_step_2.png'
)

missing_df_step_3 = f.fill_until_limit(
  df=missing_df_step_1,
  agg_dict={'value': 'mean'},
  weights=sal.weights.Queen.from_dataframe(missing_df_step_1)
)

f.plot_example(
  df=missing_df_step_3,
  fname='missing_df_step_3.png'
)

clustered_df_step_1 = test_f.gen_simulated_sectors(4, 'resident_cnt')
clustered_df_step_1['cluster'] = [
  'A', 'B', 'B', 'C',
  'A', 'B', 'B', 'B',
  'A', 'A', 'B', 'B',
  'A', 'A', 'A', 'A',
]

f.plot_example(
  df=clustered_df_step_1,
  fname='clustered_df_step_1.png',
  annotation='cluster',
  categorical=False,
)

clustered_df_step_2 = f.recluster_small_clusters(
  df=clustered_df_step_1,
  min_residents_per_cluster=15,
  weights=sal.weights.Queen.from_dataframe(clustered_df_step_1)
)

f.plot_example(
  df=clustered_df_step_2,
  fname='clustered_df_step_2.png',
  annotation='cluster',
  categorical=False,
)
