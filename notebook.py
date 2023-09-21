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
import numpy as np
import pandas as pd
import geopandas as gpd

import matplotlib.pyplot as plt
from libpysal.weights import Queen
from sklearn.preprocessing import robust_scale
from sklearn.cluster import KMeans, AgglomerativeClustering

np.random.seed(0)

# %%
INITIAL_CLUSTERS_COUNT = 200
MIN_RESIDENTS_PER_CLUSTER = 30000

# %%
grid = gpd.read_file('data/sp_setores_censitarios/35SEE250GC_SIR.shp')
grid = grid[grid['NM_MUNICIP'] == 'SÃO PAULO']
grid = grid[['CD_GEOCODI', 'geometry']]
grid.rename(columns={'CD_GEOCODI': 'id'}, inplace=True)
grid['id'] = grid['id'].astype(int)

grid_data = pd.read_csv(
    'data/Basico_SP1.csv',
    encoding='latin_1', 
    sep=';',
    decimal=','
)
grid_data = grid_data[['Cod_setor', 'Situacao_setor', 'V001', 'V002', 'V003', 'V005']]
grid_data.rename(columns={
    'Cod_setor': 'id',
    'Situacao_setor': 'sector_type',
    'V001': 'house_cnt',
    'V002': 'resident_cnt',
    'V003': 'resident_avg',
    'V005': 'income_avg'
}, inplace=True)
grid_data['id'] = grid_data['id'].astype(int)

df = grid.merge(grid_data, on='id', how='left')
weights = Queen.from_dataframe(df)


# %%
def fill_with_neighbors_data(df, agg_dict):
    sector = []
    neighbor = []
    
    for neighbors in df[df.isna().any(axis=1)].index:
        sector.extend(np.full(shape=len(weights.neighbor_offsets[neighbors]), fill_value=neighbors))
        neighbor.extend(weights.neighbor_offsets[neighbors])
    
    filler = pd.DataFrame({
        'sector': sector,
        'neighbor': neighbor
    })
    
    filler = filler.merge(
        pd.DataFrame(df.iloc[:,2:]).reset_index().rename(columns={'index': 'neighbor'}),
        on='neighbor',
        how='left'
    )
    
    filler = filler.groupby('sector', dropna=True).agg(agg_dict)
    return df.fillna(filler)


# %%
def fill_until_limit(df, agg_dict):
    null_before = 0
    null_after = 1
    
    while null_before != null_after:
        null_before = len(df[df.isna().any(axis=1)])
        df = fill_with_neighbors_data(df, agg_dict)
        null_after = len(df[df.isna().any(axis=1)])

    return df


# %%
sector_data = {
    'sector_type': lambda x: (pd.Series.mode(x)).iloc[0] if len(pd.Series.mode(x)) > 0 else x.iloc[0], 
    'house_cnt': 'mean',
    'resident_cnt': 'mean',
    'resident_avg': 'mean',
    'income_avg': 'mean',
}

df = fill_until_limit(df, sector_data)
df['house_cnt'] = df['house_cnt'].round(0)
missing_sectors = df[df.isna().any(axis=1)]

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
weights = Queen.from_dataframe(df)

# %%
cluster_variables = [
    'sector_type',
    'house_cnt',
    'resident_cnt',
    'resident_avg',
    'income_avg'
]

scaled_df = robust_scale(df[cluster_variables])
model = AgglomerativeClustering(
    linkage="ward",
    connectivity=weights.sparse,
    n_clusters=INITIAL_CLUSTERS_COUNT
)
model.fit(scaled_df)
df["cluster"] = model.labels_

# %%
residents_by_cluster = df.groupby('cluster')['resident_cnt'].sum()
small_clusters = residents_by_cluster[residents_by_cluster <= MIN_RESIDENTS_PER_CLUSTER].index

while len(small_clusters) > 0:
    df.loc[df['cluster'].isin(small_clusters), 'cluster'] = np.nan
    cluster_data = {
        'cluster': lambda x: (pd.Series.mode(x)).iloc[0] if len(pd.Series.mode(x)) > 0 else x.iloc[0], 
    }
    df = fill_until_limit(df, cluster_data)

    residents_by_cluster = df.groupby('cluster')['resident_cnt'].sum()
    small_clusters = residents_by_cluster[residents_by_cluster <= MIN_RESIDENTS_PER_CLUSTER].index

# %%
missing_sectors['cluster'] = [8.0, 0.0, 0.0]
missing_sectors = missing_sectors[['id', 'geometry', 'cluster']]
df = pd.concat([df[['id', 'geometry', 'cluster']], missing_sectors])
dissolved_df = df.dissolve(by='cluster').reset_index()

f, ax = plt.subplots(1, figsize=(9, 9))

dissolved_df.plot(
    column="cluster",
    categorical=True,
    legend=False,
    linewidth=0.3,
    ax=ax,
    edgecolor='black',
)
ax.set_axis_off()
plt.show()
