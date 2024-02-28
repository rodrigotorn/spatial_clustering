import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import Polygon
import libpysal as sal
import src.functions as f


@pytest.fixture
def simulated_data_with_missing_values():
  missing_df = gen_simulated_sectors(4)
  for i in [2, 3, 6, 7, 13]:
    missing_df.loc[i, 'value'] = np.nan
  return missing_df


def gen_simulated_sectors(n, col_name='value'):
  id_lst = [x for x in range(0, n*n)]
  geometry_lst = []
  value_lst = []

  for i in range(0, n):
    for j in range(0, n):
      geometry_lst.append(
        Polygon(((j, i), (j, i+1), (j+1, i+1), (j+1, i), (j, i)))
      )
      value_lst.append(10*(i) + 10*(n-j))

  gdf = gpd.GeoDataFrame(
    data=pd.DataFrame(
      data={
        col_name: value_lst,
        'geometry': geometry_lst
      },
      index=id_lst
    ),
    crs='EPSG:4674',
    geometry='geometry'
  )
  return gdf


def test_gen_simulated_sectors():
  expected_df = gpd.GeoDataFrame(
    data=pd.DataFrame(
      data={
        'value': [
          30, 20, 10,
          40, 30, 20,
          50, 40, 30
        ],
        'geometry': [
          Polygon(((0., 0.), (0., 1.), (1., 1.), (1., 0.), (0., 0.))),
          Polygon(((1., 0.), (1., 1.), (2., 1.), (2., 0.), (1., 0.))),
          Polygon(((2., 0.), (2., 1.), (3., 1.), (3., 0.), (2., 0.))),
          Polygon(((0., 1.), (0., 2.), (1., 2.), (1., 1.), (0., 1.))),
          Polygon(((1., 1.), (1., 2.), (2., 2.), (2., 1.), (1., 1.))),
          Polygon(((2., 1.), (2., 2.), (3., 2.), (3., 1.), (2., 1.))),
          Polygon(((0., 2.), (0., 3.), (1., 3.), (1., 2.), (0., 2.))),
          Polygon(((1., 2.), (1., 3.), (2., 3.), (2., 2.), (1., 2.))),
          Polygon(((2., 2.), (2., 3.), (3., 3.), (3., 2.), (2., 2.))),
        ]
      },
      index=[x for x in range(0, 9)]
    ),
    crs='EPSG:4674',
    geometry='geometry'
  )

  actual_df = gen_simulated_sectors(3)
  pd.testing.assert_frame_equal(
    actual_df,
    expected_df
  )


def test_fill_with_neighbors_data(simulated_data_with_missing_values):
  actual_df = f.fill_with_neighbors_data(
    df=simulated_data_with_missing_values,
    agg_dict={'value': 'mean'},
    weights=sal.weights.Queen.from_dataframe(simulated_data_with_missing_values)
  )

  expected_df = simulated_data_with_missing_values
  expected_df.loc[2, 'value'] = 35
  expected_df.loc[6, 'value'] = 38
  expected_df.loc[7, 'value'] = 35
  expected_df.loc[13, 'value'] = 54

  pd.testing.assert_frame_equal(
    actual_df,
    expected_df
  )


def test_fill_until_limit(simulated_data_with_missing_values):
  actual_df = f.fill_until_limit(
    df=simulated_data_with_missing_values,
    agg_dict={'value': 'mean'},
    weights=sal.weights.Queen.from_dataframe(simulated_data_with_missing_values)
  )

  expected_df = simulated_data_with_missing_values
  expected_df.loc[2, 'value'] = 35
  expected_df.loc[3, 'value'] = 36
  expected_df.loc[6, 'value'] = 38
  expected_df.loc[7, 'value'] = 35
  expected_df.loc[13, 'value'] = 54

  pd.testing.assert_frame_equal(
    actual_df,
    expected_df
  )


def test_recluster_small_clusters():
  clustered_df = gen_simulated_sectors(4, 'resident_cnt')
  clustered_df['cluster'] = [
    1., 2., 2., 3.,
    1., 2., 2., 2.,
    1., 1., 2., 2.,
    1., 1., 1., 1.,
  ]
  actual_df = f.recluster_small_clusters(
    df=clustered_df,
    min_residents_per_cluster=15,
    weights=sal.weights.Queen.from_dataframe(clustered_df)
  )
  expected_df = clustered_df
  expected_df['cluster'] = [
    1., 2., 2., 2.,
    1., 2., 2., 2.,
    1., 1., 2., 2.,
    1., 1., 1., 1.,
  ]

  pd.testing.assert_frame_equal(
    actual_df,
    expected_df
  )
