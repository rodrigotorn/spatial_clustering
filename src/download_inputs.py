import os
import shutil
import requests
from zipfile import ZipFile 
import warnings
warnings.filterwarnings('ignore')


import logging
from logger import get_logger
logger = get_logger('src.download_inputs', logging.INFO)


SP_GEOGRAPHIC_DATA_URL = 'https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/malhas_de_setores_censitarios__divisoes_intramunicipais/censo_2010/setores_censitarios_shp/sp/sp_setores_censitarios.zip'
SP_GEOGRAPHIC_DATA_PATH = os.getcwd() + '/data/geographic_data'

SP_DEMOGRAPHIC_DATA_URL = 'https://ftp.ibge.gov.br/Censos/Censo_Demografico_2010/Resultados_do_Universo/Agregados_por_Setores_Censitarios/SP_Capital_20231030.zip'
SP_DEMOGRAPHIC_DATA_PATH = os.getcwd() + '/data/demographic_data'
SP_DEMOGRAPHIC_DATA_FILE = '/Base informaçoes setores2010 universo SP_Capital/CSV/Basico_SP1.csv'

DF_GEOGRAPHIC_DATA_URL = 'https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/malhas_de_setores_censitarios__divisoes_intramunicipais/censo_2010/setores_censitarios_shp/df/df_setores_censitarios.zip'
DF_GEOGRAPHIC_DATA_PATH = os.getcwd() + '/data/geographic_data'

DF_DEMOGRAPHIC_DATA_URL = 'https://ftp.ibge.gov.br/Censos/Censo_Demografico_2010/Resultados_do_Universo/Agregados_por_Setores_Censitarios/DF_20231030.zip'
DF_DEMOGRAPHIC_DATA_PATH = os.getcwd() + '/data/demographic_data'
DF_DEMOGRAPHIC_DATA_FILE = '/Base informaçoes setores2010 universo DF//CSV/Basico_DF.csv'


def download_file(url: str, path: str) -> None:
	r = requests.get(url, allow_redirects=True)
	open(path, 'wb').write(r.content)

def extract_file(zip_path: str, extraction_path: str) -> None:
	with ZipFile(zip_path, 'r') as zObject:
		zObject.extractall(path=extraction_path)
	zObject.close() 

if __name__ == '__main__':
	logger.info(f'Downloading SP geographic data into {SP_GEOGRAPHIC_DATA_PATH}')
	download_file(SP_GEOGRAPHIC_DATA_URL, SP_GEOGRAPHIC_DATA_PATH + '.zip')
	extract_file(SP_GEOGRAPHIC_DATA_PATH + '.zip', SP_GEOGRAPHIC_DATA_PATH)
	os.remove(SP_GEOGRAPHIC_DATA_PATH + '.zip')

	logger.info(f'Downloading SP demographic data into {SP_DEMOGRAPHIC_DATA_PATH}')
	download_file(SP_DEMOGRAPHIC_DATA_URL, SP_DEMOGRAPHIC_DATA_PATH + '.zip')
	extract_file(SP_DEMOGRAPHIC_DATA_PATH + '.zip', SP_DEMOGRAPHIC_DATA_PATH)
	shutil.copyfile(SP_DEMOGRAPHIC_DATA_PATH + SP_DEMOGRAPHIC_DATA_FILE, SP_DEMOGRAPHIC_DATA_PATH + '/Basico_SP1.csv')
	os.remove(SP_DEMOGRAPHIC_DATA_PATH + '.zip')
	shutil.rmtree(SP_DEMOGRAPHIC_DATA_PATH + '/Base informaçoes setores2010 universo SP_Capital/')

	
	logger.info(f'Downloading DF geographic data into {DF_GEOGRAPHIC_DATA_PATH}')
	download_file(DF_GEOGRAPHIC_DATA_URL, DF_GEOGRAPHIC_DATA_PATH + '.zip')
	extract_file(DF_GEOGRAPHIC_DATA_PATH + '.zip', DF_GEOGRAPHIC_DATA_PATH)
	os.remove(DF_GEOGRAPHIC_DATA_PATH + '.zip')

	logger.info(f'Downloading DF demographic data into {DF_DEMOGRAPHIC_DATA_PATH}')
	download_file(DF_DEMOGRAPHIC_DATA_URL, DF_DEMOGRAPHIC_DATA_PATH + '.zip')
	extract_file(DF_DEMOGRAPHIC_DATA_PATH + '.zip', DF_DEMOGRAPHIC_DATA_PATH)
	shutil.copyfile(DF_DEMOGRAPHIC_DATA_PATH + DF_DEMOGRAPHIC_DATA_FILE, DF_DEMOGRAPHIC_DATA_PATH + '/Basico_DF.csv')
	os.remove(DF_DEMOGRAPHIC_DATA_PATH + '.zip')
	shutil.rmtree(DF_DEMOGRAPHIC_DATA_PATH + '/Base informaçoes setores2010 universo DF/')
