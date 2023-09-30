import numpy as np
import pandas as pd
import geopandas as gpd
import libpysal as sal
import matplotlib.pyplot as plt
from sklearn.preprocessing import robust_scale
from sklearn.cluster import KMeans, AgglomerativeClustering


def read_sp_geographic_data(path: str) -> gpd.GeoDataFrame:
  gdf: gpd.GeoDataFrame = gpd.read_file(path)
  gdf = gdf[gdf['NM_MUNICIP'] == 'SÃO PAULO']
  gdf = gdf[['CD_GEOCODI', 'geometry']]
  gdf.rename(columns={'CD_GEOCODI': 'id'}, inplace=True)
  gdf['id'] = gdf['id'].astype(int)
  return gdf

def read_sp_demographic_data(path: str) -> pd.DataFrame:
  df: pd.DataFrame = pd.read_csv(
      path,
      encoding='latin_1',
      sep=';',
      decimal=','
  )
  df = df[[
    'Cod_setor',
    'Situacao_setor',
    'V001',
    'V002',
    'V003',
    'V005'
  ]]
  df.rename(columns={
    'Cod_setor': 'id',
    'Situacao_setor': 'sector_type',
    'V001': 'house_cnt',
    'V002': 'resident_cnt',
    'V003': 'resident_avg',
    'V005': 'income_avg'
  }, inplace=True)
  df['id'] = df['id'].astype(int)
  return df

def fill_with_neighbors_data(
    df: pd.DataFrame,
    agg_dict: dict,
    weights: sal.weights.Queen
  ) -> pd.DataFrame:
  sector: list = []
  neighbor: list = []

  for neighbors in df[df.isna().any(axis=1)].index:
    sector.extend(
      np.full(
        shape=len(weights.neighbor_offsets[neighbors]),
        fill_value=neighbors
      )
    )
    neighbor.extend(weights.neighbor_offsets[neighbors])

  filler = pd.DataFrame({
    'sector': sector,
    'neighbor': neighbor
  })
  filler = filler.merge(
    pd.DataFrame(
      df.iloc[:,2:]).reset_index().rename(
        columns={'index': 'neighbor'}
    ),
    on='neighbor',
    how='left'
  )
  filler = filler.groupby('sector', dropna=True).agg(agg_dict)
  return df.fillna(filler)

def fill_until_limit(
    df: pd.DataFrame,
    agg_dict: dict,
    weights: sal.weights.Queen
  ) -> pd.DataFrame:
  null_before: int  = 0
  null_after: int = 1

  while null_before != null_after:
    null_before = len(df[df.isna().any(axis=1)])
    df = fill_with_neighbors_data(df, agg_dict, weights)
    null_after = len(df[df.isna().any(axis=1)])

  return df

def fill_missing_data(
    df: pd.DataFrame,
    weights: sal.weights.Queen
  ) -> pd.DataFrame:
  sector_data: dict = {
    'sector_type': lambda x:
      (pd.Series.mode(x)).iloc[0] if len(pd.Series.mode(x)) > 0 else x.iloc[0],
    'house_cnt': 'mean',
    'resident_cnt': 'mean',
    'resident_avg': 'mean',
    'income_avg': 'mean',
  }
  df = fill_until_limit(df, sector_data, weights)
  df['house_cnt'] = df['house_cnt'].round(0)
  missing_sectors: pd.DataFrame = df[df.isna().any(axis=1)]
  df.dropna(inplace=True)
  df.reset_index(drop=True, inplace=True)
  return df, missing_sectors

def agglomerative_cluster(
  df: pd.DataFrame,
  initial_cluster_count: int,
  weights: sal.weights.Queen
  ) -> pd.DataFrame:
  cluster_variables: list = [
    'sector_type',
    'house_cnt',
    'resident_cnt',
    'resident_avg',
    'income_avg'
  ]
  scaled_df: pd.DataFrame = robust_scale(df[cluster_variables])
  model: sklearn.cluster.AgglomerativeClustering = AgglomerativeClustering(
    linkage='ward',
    connectivity=weights.sparse,
    n_clusters=initial_cluster_count
  )
  model.fit(scaled_df)
  df['cluster'] = model.labels_
  return df

def recluster_small_clusters(
  df: pd.DataFrame,
  min_residents_per_cluster: int,
  weights: sal.weights.Queen
  ) -> pd.DataFrame:
  residents_by_cluster: pd.DataFrame.GroupBy = \
    df.groupby('cluster')['resident_cnt'].sum()
  small_clusters: pd.DataFrame.index = \
    residents_by_cluster[residents_by_cluster <= min_residents_per_cluster].index

  while len(small_clusters) > 0:
    df.loc[df['cluster'].isin(small_clusters), 'cluster'] = np.nan
    cluster_data: dict = {
        'cluster': lambda x:
          (pd.Series.mode(x)).iloc[0] if len(pd.Series.mode(x)) > 0 else x.iloc[0],
    }
    df = fill_until_limit(df, cluster_data, weights)
    residents_by_cluster = df.groupby('cluster')['resident_cnt'].sum()
    small_clusters = \
      residents_by_cluster[residents_by_cluster <= min_residents_per_cluster].index

  return df

def manually_fill_remaining_sectors(
  df: pd.DataFrame,
  missing_sectors: pd.DataFrame
  ) -> pd.DataFrame:
  missing_sectors['cluster'] = [8.0, 0.0, 0.0]
  missing_sectors = missing_sectors[['id', 'geometry', 'cluster']]
  df = pd.concat([df[['id', 'geometry', 'cluster']], missing_sectors])
  return df

def plot_regions(df: pd.DataFrame) -> None:
  f, ax = plt.subplots(1, figsize=(9, 9))

  df.plot(
    column='cluster',
    categorical=True,
    legend=False,
    linewidth=0.3,
    ax=ax,
    edgecolor='black',
  )
  ax.set_axis_off()
  return plt.show()

if __name__ == '__main__':
  pass
