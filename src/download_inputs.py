import os
import shutil
import requests
from zipfile import ZipFile 
import warnings
warnings.filterwarnings('ignore')


import logging
from logger import get_logger
logger = get_logger('src.download_inputs', logging.INFO)


GEOGRAPHIC_DATA_URL = 'https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/malhas_de_setores_censitarios__divisoes_intramunicipais/censo_2010/setores_censitarios_shp/sp/sp_setores_censitarios.zip'
GEOGRAPHIC_DATA_PATH = os.getcwd() + '/data/geographic_data'

DEMOGRAPHIC_DATA_URL = 'https://ftp.ibge.gov.br/Censos/Censo_Demografico_2010/Resultados_do_Universo/Agregados_por_Setores_Censitarios/SP_Capital_20231030.zip'
DEMOGRAPHIC_DATA_PATH = os.getcwd() + '/data/demographic_data'
DEMOGRAPHIC_DATA_FILE = '/Base informaçoes setores2010 universo SP_Capital/CSV/Basico_SP1.csv'


def download_file(url: str, path: str) -> None:
	r = requests.get(url, allow_redirects=True)
	open(path, 'wb').write(r.content)

def extract_file(zip_path: str, extraction_path: str) -> None:
	with ZipFile(zip_path, 'r') as zObject:
		zObject.extractall(path=extraction_path)
	zObject.close() 

if __name__ == '__main__':
	logger.info(f'Downloading geopraphic data into {GEOGRAPHIC_DATA_PATH}')
	download_file(GEOGRAPHIC_DATA_URL, GEOGRAPHIC_DATA_PATH + '.zip')
	extract_file(GEOGRAPHIC_DATA_PATH + '.zip', GEOGRAPHIC_DATA_PATH)
	os.remove(GEOGRAPHIC_DATA_PATH + '.zip')


	logger.info(f'Downloading demoraphic data into {DEMOGRAPHIC_DATA_PATH}')
	download_file(DEMOGRAPHIC_DATA_URL, DEMOGRAPHIC_DATA_PATH + '.zip')
	extract_file(DEMOGRAPHIC_DATA_PATH + '.zip', DEMOGRAPHIC_DATA_PATH)
	shutil.copyfile(DEMOGRAPHIC_DATA_PATH + DEMOGRAPHIC_DATA_FILE, DEMOGRAPHIC_DATA_PATH + '/Basico_SP1.csv')
	os.remove(DEMOGRAPHIC_DATA_PATH + '.zip')
	shutil.rmtree(DEMOGRAPHIC_DATA_PATH + '/Base informaçoes setores2010 universo SP_Capital/')
