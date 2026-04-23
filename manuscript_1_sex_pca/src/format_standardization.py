### loads original raw data


import pandas as pd
import numpy as np
from datetime import datetime
import os

os.makedirs('standardized/',exist_ok=True)
y1 = np.array([2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021])

def standardize_years(fraction):
	# for year in range(2000,2003):
	# 	time1 = datetime.now()
	# 	df = pd.read_parquet(f'MS_data/{year}_{fraction}.parquet')
	# 	df.columns = df.columns.str.upper()
	# 	time2 = datetime.now()-time1
	# 	print(f'{year}, load time: {time2}')

	# 	df['CIGS'] = df['TOBACCO']
	# 	df.to_parquet(f'standardized/{year}_{fraction}.parquet')

	for year in range(2010,2014):
		time1 = datetime.now()
		df = pd.read_parquet(f'MS_data/{year}_{fraction}.parquet')
		df.columns = df.columns.str.upper()

		df.loc[df.index[df['CIG_1'].isna()], 'CIG_1'] =99
		df.loc[df.index[df['CIG_2'].isna()], 'CIG_2'] =99
		df.loc[df.index[df['CIG_3'].isna()], 'CIG_3'] =99
		df['CIG_1'] = df['CIG_1'].astype(int)
		df['CIG_2'] = df['CIG_2'].astype(int)
		df['CIG_3'] = df['CIG_3'].astype(int)

		time2 = datetime.now()-time1
		print(f'{year}, load time: {time2}')

		df['CIGS'] = 2
		df.loc[df.index[df['CIG_1']>0],'CIGS'] = 1
		df.loc[df.index[df['CIG_2']>0],'CIGS'] = 1
		df.loc[df.index[df['CIG_3']>0],'CIGS'] = 1

		df.loc[df.index[df['CIG_1']==99],'CIGS'] = 9
		df.loc[df.index[df['CIG_2']==99],'CIGS'] = 9
		df.loc[df.index[df['CIG_3']==99],'CIGS'] = 9


		df.loc[df.index[df['TOBUSE']==1], 'CIGS'] = 1
		df.loc[df.index[df['CIG_REC']=='Y'], 'CIGS'] = 1

		df['DOD_YY'] = df['DTHYR']
		df['DOD_MM'] = df['DTHMON']
		df['DWEEKDAY'] = df['WEEKDAYD']

		df['CA_MNSB'] = df['CA_MENIN']
		df['CA_CCHD'] = df['CA_HEART']
		df['CA_CDH'] = df['CA_HERNIA']
		df['CA_OMPH'] = df['CA_OMPHA']
		df['CA_GAST'] = df['CA_GASTRO']
		df['CA_CLEFT']= df['CA_CLEFTLP']
		df['CA_CLPAL']= df['CA_CLEFT']
		df['CA_DOWN'] = df['CA_DOWNS']
		df['CA_DISOR']= df['CA_CHROM']
		df['CA_HYPO'] = df['CA_HYPOS']

		df['AB_AVEN1'] = df['AB_VENT']
		df['AB_AVEN6'] = df['AB_VENT6']
		df['AB_SURF'] = df['AB_SURFAC']
		df['AB_ANTI'] = df['AB_ANTIBIO']

		df.to_parquet(f'standardized/{year}_{fraction}.parquet')


	for year in range(2014,2016):
		time1 = datetime.now()
		df = pd.read_parquet(f'MS_data/{year}_{fraction}.parquet')
		df.columns = df.columns.str.upper()

		time2 = datetime.now()-time1
		print(f'{year}, load time: {time2}')

		df['PRECARE'] = df['RECARE']

		df.to_parquet(f'standardized/{year}_{fraction}.parquet')


	# for year in range(2014,2018):
	# 	time1 = datetime.now()
	# 	df = pd.read_parquet(f'MS_data/{year}_{fraction}.parquet')
	# 	df.columns = df.columns.str.upper()

	# 	df.loc[df.index[df['CIG_0'].isna()], 'CIG_0'] =99
	# 	df.loc[df.index[df['CIG_1'].isna()], 'CIG_1'] =99
	# 	df.loc[df.index[df['CIG_2'].isna()], 'CIG_2'] =99
	# 	df.loc[df.index[df['CIG_3'].isna()], 'CIG_3'] =99
	# 	df['CIG_0'] = df['CIG_0'].astype(int)
	# 	df['CIG_1'] = df['CIG_1'].astype(int)
	# 	df['CIG_2'] = df['CIG_2'].astype(int)
	# 	df['CIG_3'] = df['CIG_3'].astype(int)


	# 	############## pre-pregnancy and gestional smoking
	# 	df['CIGS'] = 2
	# 	df.loc[df.index[df['CIG_0']>0],'CIGS'] = 1
	# 	df.loc[df.index[df['CIG_1']>0],'CIGS'] = 1
	# 	df.loc[df.index[df['CIG_2']>0],'CIGS'] = 1
	# 	df.loc[df.index[df['CIG_3']>0],'CIGS'] = 1

	# 	df.loc[df.index[df['CIG_1']==99],'CIGS'] = 9
	# 	df.loc[df.index[df['CIG_2']==99],'CIGS'] = 9
	# 	df.loc[df.index[df['CIG_3']==99],'CIGS'] = 9

	# 	############## pre-pregnancy smoking
	# 	df['PRE_CIGS'] = 2
	# 	df.loc[df.index[df['CIG_0']>0],'PRE_CIGS'] = 1
	# 	df.loc[df.index[df['CIG_1']>0],'PRE_CIGS'] = 1
	# 	df.loc[df.index[df['CIG_2']>0],'PRE_CIGS'] = 1
	# 	df.loc[df.index[df['CIG_3']>0],'PRE_CIGS'] = 1

	# 	df.loc[df.index[df['CIG_1']==99],'PRE_CIGS'] = 9
	# 	df.loc[df.index[df['CIG_2']==99],'PRE_CIGS'] = 9
	# 	df.loc[df.index[df['CIG_3']==99],'PRE_CIGS'] = 9

	# 	############## gestational smoking
	# 	df['GEST_CIGS'] = 2
	# 	df.loc[df.index[df['CIG_0']>0],'GEST_CIGS'] = 1
	# 	df.loc[df.index[df['CIG_1']>0],'GEST_CIGS'] = 1
	# 	df.loc[df.index[df['CIG_2']>0],'GEST_CIGS'] = 1
	# 	df.loc[df.index[df['CIG_3']>0],'GEST_CIGS'] = 1

	# 	df.loc[df.index[df['CIG_1']==99],'GEST_CIGS'] = 9
	# 	df.loc[df.index[df['CIG_2']==99],'GEST_CIGS'] = 9
	# 	df.loc[df.index[df['CIG_3']==99],'GEST_CIGS'] = 9

	# 	df['AGEDX'] = df['AGED']
	# 	# df.loc[df.index[df['TOBUSE']==1], 'CIGS'] = 1
	# 	df.loc[df.index[df['CIG_REC']=='Y'], 'CIGS'] = 1

	# 	df.to_parquet(f'standardized/{year}_{fraction}_numerator.parquet')

	# 	time2 = datetime.now()-time1
	# 	print(f'{year}, work time: {time2}')

	# for year in range(2018,2022):
	# 	time1 = datetime.now()
	# 	df = pd.read_parquet(f'MS_data/{year}_{fraction}.parquet')
	# 	df.columns = df.columns.str.upper()

	# 	df.loc[df.index[df['CIG_0'].isna()], 'CIG_0'] =99
	# 	df.loc[df.index[df['CIG_1'].isna()], 'CIG_1'] =99
	# 	df.loc[df.index[df['CIG_2'].isna()], 'CIG_2'] =99
	# 	df.loc[df.index[df['CIG_3'].isna()], 'CIG_3'] =99
	# 	df['CIG_0'] = df['CIG_0'].astype(int)
	# 	df['CIG_1'] = df['CIG_1'].astype(int)
	# 	df['CIG_2'] = df['CIG_2'].astype(int)
	# 	df['CIG_3'] = df['CIG_3'].astype(int)

	# 	df['AGED'] = df['AGEDX'].copy()


	# 	############## pre-pregnancy and gestional smoking
	# 	df['CIGS'] = 2
	# 	df.loc[df.index[df['CIG_0']>0],'CIGS'] = 1
	# 	df.loc[df.index[df['CIG_1']>0],'CIGS'] = 1
	# 	df.loc[df.index[df['CIG_2']>0],'CIGS'] = 1
	# 	df.loc[df.index[df['CIG_3']>0],'CIGS'] = 1

	# 	df.loc[df.index[df['CIG_1']==99],'CIGS'] = 9
	# 	df.loc[df.index[df['CIG_2']==99],'CIGS'] = 9
	# 	df.loc[df.index[df['CIG_3']==99],'CIGS'] = 9

	# 	############## pre-pregnancy smoking
	# 	df['PRE_CIGS'] = 2
	# 	df.loc[df.index[df['CIG_0']>0],'PRE_CIGS'] = 1
	# 	df.loc[df.index[df['CIG_1']>0],'PRE_CIGS'] = 1
	# 	df.loc[df.index[df['CIG_2']>0],'PRE_CIGS'] = 1
	# 	df.loc[df.index[df['CIG_3']>0],'PRE_CIGS'] = 1

	# 	df.loc[df.index[df['CIG_1']==99],'PRE_CIGS'] = 9
	# 	df.loc[df.index[df['CIG_2']==99],'PRE_CIGS'] = 9
	# 	df.loc[df.index[df['CIG_3']==99],'PRE_CIGS'] = 9

	# 	############## gestational smoking
	# 	df['GEST_CIGS'] = 2
	# 	df.loc[df.index[df['CIG_0']>0],'GEST_CIGS'] = 1
	# 	df.loc[df.index[df['CIG_1']>0],'GEST_CIGS'] = 1
	# 	df.loc[df.index[df['CIG_2']>0],'GEST_CIGS'] = 1
	# 	df.loc[df.index[df['CIG_3']>0],'GEST_CIGS'] = 1

	# 	df.loc[df.index[df['CIG_1']==99],'GEST_CIGS'] = 9
	# 	df.loc[df.index[df['CIG_2']==99],'GEST_CIGS'] = 9
	# 	df.loc[df.index[df['CIG_3']==99],'GEST_CIGS'] = 9

	# 	# df.loc[df.index[df['TOBUSE']==1], 'CIGS'] = 1
	# 	df.loc[df.index[df['CIG_REC']=='Y'], 'CIGS'] = 1

	# 	df.to_parquet(f'standardized/{year}_{fraction}_denominator.parquet')

	# 	time2 = datetime.now()-time1
	# 	print(f'{year}, work time: {time2}')

def standardize_years_den(fraction):
	# for year in range(2000,2003):
	# 	time1 = datetime.now()
	# 	df = pd.read_parquet(f'MS_data/{year}_{fraction}.parquet')
	# 	df.columns = df.columns.str.upper()
	# 	time2 = datetime.now()-time1
	# 	print(f'{year}, load time: {time2}')

	# 	df['CIGS'] = df['TOBACCO']
	# 	df.to_parquet(f'standardized/{year}_{fraction}.parquet')

	# for year in range(2003,2014):
	# 	time1 = datetime.now()
	# 	df = pd.read_parquet(f'MS_data/{year}_{fraction}.parquet')
	# 	df.columns = df.columns.str.upper()

	# 	df.loc[df.index[df['CIG_1'].isna()], 'CIG_1'] =99
	# 	df.loc[df.index[df['CIG_2'].isna()], 'CIG_2'] =99
	# 	df.loc[df.index[df['CIG_3'].isna()], 'CIG_3'] =99
	# 	df['CIG_1'] = df['CIG_1'].astype(int)
	# 	df['CIG_2'] = df['CIG_2'].astype(int)
	# 	df['CIG_3'] = df['CIG_3'].astype(int)

	# 	time2 = datetime.now()-time1
	# 	print(f'{year}, load time: {time2}')

	# 	df['CIGS'] = 2
	# 	df.loc[df.index[df['CIG_1']>0],'CIGS'] = 1
	# 	df.loc[df.index[df['CIG_2']>0],'CIGS'] = 1
	# 	df.loc[df.index[df['CIG_3']>0],'CIGS'] = 1

	# 	df.loc[df.index[df['CIG_1']==99],'CIGS'] = 9
	# 	df.loc[df.index[df['CIG_2']==99],'CIGS'] = 9
	# 	df.loc[df.index[df['CIG_3']==99],'CIGS'] = 9


	# 	df.loc[df.index[df['TOBUSE']==1], 'CIGS'] = 1
	# 	df.loc[df.index[df['CIG_REC']=='Y'], 'CIGS'] = 1

	# 	# df['DOD_YY'] = df['DTHYR']
	# 	df.to_parquet(f'standardized/{year}_{fraction}.parquet')

	# for year in range(2014,2018):
	# 	time1 = datetime.now()
	# 	df = pd.read_parquet(f'MS_data/{year}_{fraction}.parquet')
	# 	df.columns = df.columns.str.upper()

	# 	df.loc[df.index[df['CIG_0'].isna()], 'CIG_0'] =99
	# 	df.loc[df.index[df['CIG_1'].isna()], 'CIG_1'] =99
	# 	df.loc[df.index[df['CIG_2'].isna()], 'CIG_2'] =99
	# 	df.loc[df.index[df['CIG_3'].isna()], 'CIG_3'] =99
	# 	df['CIG_0'] = df['CIG_0'].astype(int)
	# 	df['CIG_1'] = df['CIG_1'].astype(int)
	# 	df['CIG_2'] = df['CIG_2'].astype(int)
	# 	df['CIG_3'] = df['CIG_3'].astype(int)


	# 	############## pre-pregnancy and gestional smoking
	# 	df['CIGS'] = 2
	# 	df.loc[df.index[df['CIG_0']>0],'CIGS'] = 1
	# 	df.loc[df.index[df['CIG_1']>0],'CIGS'] = 1
	# 	df.loc[df.index[df['CIG_2']>0],'CIGS'] = 1
	# 	df.loc[df.index[df['CIG_3']>0],'CIGS'] = 1

	# 	df.loc[df.index[df['CIG_1']==99],'CIGS'] = 9
	# 	df.loc[df.index[df['CIG_2']==99],'CIGS'] = 9
	# 	df.loc[df.index[df['CIG_3']==99],'CIGS'] = 9

	# 	############## pre-pregnancy smoking
	# 	df['PRE_CIGS'] = 2
	# 	df.loc[df.index[df['CIG_0']>0],'PRE_CIGS'] = 1
	# 	df.loc[df.index[df['CIG_1']>0],'PRE_CIGS'] = 1
	# 	df.loc[df.index[df['CIG_2']>0],'PRE_CIGS'] = 1
	# 	df.loc[df.index[df['CIG_3']>0],'PRE_CIGS'] = 1

	# 	df.loc[df.index[df['CIG_1']==99],'PRE_CIGS'] = 9
	# 	df.loc[df.index[df['CIG_2']==99],'PRE_CIGS'] = 9
	# 	df.loc[df.index[df['CIG_3']==99],'PRE_CIGS'] = 9

	# 	############## gestational smoking
	# 	df['GEST_CIGS'] = 2
	# 	df.loc[df.index[df['CIG_0']>0],'GEST_CIGS'] = 1
	# 	df.loc[df.index[df['CIG_1']>0],'GEST_CIGS'] = 1
	# 	df.loc[df.index[df['CIG_2']>0],'GEST_CIGS'] = 1
	# 	df.loc[df.index[df['CIG_3']>0],'GEST_CIGS'] = 1

	# 	df.loc[df.index[df['CIG_1']==99],'GEST_CIGS'] = 9
	# 	df.loc[df.index[df['CIG_2']==99],'GEST_CIGS'] = 9
	# 	df.loc[df.index[df['CIG_3']==99],'GEST_CIGS'] = 9
	# 	# df['AGEDX'] = df['AGED']
	# 	# df.loc[df.index[df['TOBUSE']==1], 'CIGS'] = 1
	# 	df.loc[df.index[df['CIG_REC']=='Y'], 'CIGS'] = 1

	# 	df.to_parquet(f'standardized/{year}_{fraction}.parquet')

	# 	time2 = datetime.now()-time1
	# 	print(f'{year}, work time: {time2}')

	for year in range(2014,2016):
		time1 = datetime.now()
		df = pd.read_parquet(f'MS_data/{year}_{fraction}.parquet')
		df.columns = df.columns.str.upper()
		# df['UCOD'] = df['UC0D']
		df['PRECARE'] = df['RECARE']

		df.to_parquet(f'standardized/{year}_{fraction}.parquet')

		time2 = datetime.now()-time1
		print(f'{year}, work time: {time2}')

	for year in range(2019,2022):
		time1 = datetime.now()
		df = pd.read_parquet(f'MS_data/{year}_{fraction}.parquet')
		df.columns = df.columns.str.upper()
		# df['UCOD'] = df['UC0D']
		df['CO_DODYY'] = df['CO_YOD']

		df.to_parquet(f'standardized/{year}_{fraction}.parquet')

		time2 = datetime.now()-time1
		print(f'{year}, work time: {time2}')

	# for year in range(2018,2022):
	# 	time1 = datetime.now()
	# 	df = pd.read_parquet(f'MS_data/{year}_{fraction}.parquet')
	# 	df.columns = df.columns.str.upper()

	# 	df.loc[df.index[df['CIG_0'].isna()], 'CIG_0'] =99
	# 	df.loc[df.index[df['CIG_1'].isna()], 'CIG_1'] =99
	# 	df.loc[df.index[df['CIG_2'].isna()], 'CIG_2'] =99
	# 	df.loc[df.index[df['CIG_3'].isna()], 'CIG_3'] =99
	# 	df['CIG_0'] = df['CIG_0'].astype(int)
	# 	df['CIG_1'] = df['CIG_1'].astype(int)
	# 	df['CIG_2'] = df['CIG_2'].astype(int)
	# 	df['CIG_3'] = df['CIG_3'].astype(int)

	# 	# df['AGED'] = df['AGEDX'].copy()

	# 	############## pre-pregnancy and gestional smoking
	# 	df['CIGS'] = 2
	# 	df.loc[df.index[df['CIG_0']>0],'CIGS'] = 1
	# 	df.loc[df.index[df['CIG_1']>0],'CIGS'] = 1
	# 	df.loc[df.index[df['CIG_2']>0],'CIGS'] = 1
	# 	df.loc[df.index[df['CIG_3']>0],'CIGS'] = 1

	# 	df.loc[df.index[df['CIG_1']==99],'CIGS'] = 9
	# 	df.loc[df.index[df['CIG_2']==99],'CIGS'] = 9
	# 	df.loc[df.index[df['CIG_3']==99],'CIGS'] = 9

	# 	############## pre-pregnancy smoking
	# 	df['PRE_CIGS'] = 2
	# 	df.loc[df.index[df['CIG_0']>0],'PRE_CIGS'] = 1
	# 	df.loc[df.index[df['CIG_1']>0],'PRE_CIGS'] = 1
	# 	df.loc[df.index[df['CIG_2']>0],'PRE_CIGS'] = 1
	# 	df.loc[df.index[df['CIG_3']>0],'PRE_CIGS'] = 1

	# 	df.loc[df.index[df['CIG_1']==99],'PRE_CIGS'] = 9
	# 	df.loc[df.index[df['CIG_2']==99],'PRE_CIGS'] = 9
	# 	df.loc[df.index[df['CIG_3']==99],'PRE_CIGS'] = 9

	# 	############## gestational smoking
	# 	df['GEST_CIGS'] = 2
	# 	df.loc[df.index[df['CIG_0']>0],'GEST_CIGS'] = 1
	# 	df.loc[df.index[df['CIG_1']>0],'GEST_CIGS'] = 1
	# 	df.loc[df.index[df['CIG_2']>0],'GEST_CIGS'] = 1
	# 	df.loc[df.index[df['CIG_3']>0],'GEST_CIGS'] = 1

	# 	df.loc[df.index[df['CIG_1']==99],'GEST_CIGS'] = 9
	# 	df.loc[df.index[df['CIG_2']==99],'GEST_CIGS'] = 9
	# 	df.loc[df.index[df['CIG_3']==99],'GEST_CIGS'] = 9

	# 	# df.loc[df.index[df['TOBUSE']==1], 'CIGS'] = 1
	# 	df.loc[df.index[df['CIG_REC']=='Y'], 'CIGS'] = 1

	# 	df.to_parquet(f'standardized/{year}_{fraction}.parquet')

	# 	time2 = datetime.now()-time1
	# 	print(f'{year}, work time: {time2}')

standardize_years_den('denominator')
standardize_years('numerator')
