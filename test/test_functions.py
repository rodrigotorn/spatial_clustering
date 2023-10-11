import pandas as pd
import geopandas as gpd
from shapely import Polygon


def gen_simulated_sectors(n):
  id_lst = [x for x in range(1, n*n+1)]
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
        'value': value_lst,
        'geometry': geometry_lst
      },
      index=id_lst
    ),
    crs='EPSG:4674',
    geometry='geometry'
  )
  return gdf
