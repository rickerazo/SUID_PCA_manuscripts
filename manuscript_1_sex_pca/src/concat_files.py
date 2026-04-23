'''
Loads *almost* raw data (preprocessed was format standardization).

Load the data from each year and systematically concatenate into a larger file

Concatenate deaths associated with specific death codes
concatenate denominator files, those than contain all US live births

concatenate survival files, those that are either not associated with a ICD death code nor a death year
Finally, concatenate all dead cases, regardless of causes of death.


Simultaneously, an important goal is to preserve as many variables as possible
since a database rich in individual cases and also in variables will be useful for the classification algorithm


MUST formulate a sound hypothesis in order to build a logistic regression model.

'''


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

import warnings

# Suppress the specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated")

os.makedirs('standardized/consolidated',exist_ok=True)
def infage_recode_weeks(df):
	df['AGED'] = df['AGED'].astype(int)
	df['INFAGE'] = 52
	df.loc[df.index[df['AGED']<364], 'INFAGE'] = 51
	df.loc[df.index[df['AGED']<357], 'INFAGE'] = 50
	df.loc[df.index[df['AGED']<350], 'INFAGE'] = 49
	df.loc[df.index[df['AGED']<343], 'INFAGE'] = 48
	df.loc[df.index[df['AGED']<336], 'INFAGE'] = 47
	df.loc[df.index[df['AGED']<329], 'INFAGE'] = 46
	df.loc[df.index[df['AGED']<322], 'INFAGE'] = 45
	df.loc[df.index[df['AGED']<315], 'INFAGE'] = 44
	df.loc[df.index[df['AGED']<308], 'INFAGE'] = 43
	df.loc[df.index[df['AGED']<301], 'INFAGE'] = 42
	df.loc[df.index[df['AGED']<294], 'INFAGE'] = 41
	df.loc[df.index[df['AGED']<287], 'INFAGE'] = 40
	df.loc[df.index[df['AGED']<280], 'INFAGE'] = 39
	df.loc[df.index[df['AGED']<273], 'INFAGE'] = 38
	df.loc[df.index[df['AGED']<266], 'INFAGE'] = 37
	df.loc[df.index[df['AGED']<259], 'INFAGE'] = 36
	df.loc[df.index[df['AGED']<252], 'INFAGE'] = 35
	df.loc[df.index[df['AGED']<245], 'INFAGE'] = 34
	df.loc[df.index[df['AGED']<238], 'INFAGE'] = 33
	df.loc[df.index[df['AGED']<231], 'INFAGE'] = 32
	df.loc[df.index[df['AGED']<224], 'INFAGE'] = 31
	df.loc[df.index[df['AGED']<217], 'INFAGE'] = 30
	df.loc[df.index[df['AGED']<210], 'INFAGE'] = 29
	df.loc[df.index[df['AGED']<203], 'INFAGE'] = 28
	df.loc[df.index[df['AGED']<196], 'INFAGE'] = 27
	df.loc[df.index[df['AGED']<189], 'INFAGE'] = 26
	df.loc[df.index[df['AGED']<182], 'INFAGE'] = 25
	df.loc[df.index[df['AGED']<175], 'INFAGE'] = 24
	df.loc[df.index[df['AGED']<168], 'INFAGE'] = 23
	df.loc[df.index[df['AGED']<161], 'INFAGE'] = 22
	df.loc[df.index[df['AGED']<154], 'INFAGE'] = 21
	df.loc[df.index[df['AGED']<147], 'INFAGE'] = 20
	df.loc[df.index[df['AGED']<140], 'INFAGE'] = 19
	df.loc[df.index[df['AGED']<133], 'INFAGE'] = 18
	df.loc[df.index[df['AGED']<126], 'INFAGE'] = 17
	df.loc[df.index[df['AGED']<119], 'INFAGE'] = 16
	df.loc[df.index[df['AGED']<112], 'INFAGE'] = 15
	df.loc[df.index[df['AGED']<105], 'INFAGE'] = 14
	df.loc[df.index[df['AGED']<98], 'INFAGE'] = 13
	df.loc[df.index[df['AGED']<91], 'INFAGE'] = 12
	df.loc[df.index[df['AGED']<84], 'INFAGE'] = 11
	df.loc[df.index[df['AGED']<77], 'INFAGE'] = 10
	df.loc[df.index[df['AGED']<70], 'INFAGE'] = 9
	df.loc[df.index[df['AGED']<63], 'INFAGE'] = 8
	df.loc[df.index[df['AGED']<56], 'INFAGE'] = 7
	df.loc[df.index[df['AGED']<49], 'INFAGE'] = 6
	df.loc[df.index[df['AGED']<42], 'INFAGE'] = 5
	df.loc[df.index[df['AGED']<35], 'INFAGE'] = 4
	df.loc[df.index[df['AGED']<28], 'INFAGE'] = 3
	df.loc[df.index[df['AGED']<21], 'INFAGE'] = 2
	df.loc[df.index[df['AGED']<14], 'INFAGE'] = 1
	df.loc[df.index[df['AGED']<7], 'INFAGE'] = 0
	# df.loc[df.index[df['AGED']<2], 'INFAGE'] = np.nan

	return df['INFAGE']

def infage_recode_days(df):
	df['AGED'] = df['AGED'].astype(int)
	df['INFAGE'] = np.nan
	df.loc[df.index[df['AGED']<=2], 'INFAGE'] = 2
	df.loc[df.index[df['AGED']<=1], 'INFAGE'] = 1

	return df['INFAGE']

def concatenate_years(icd_code,variables, cat_vars):
	out0 = pd.DataFrame()
	out1 = pd.DataFrame()
	print(f'\n{icd_code}\n')
	ratio1 = np.array([])
	fig,ax = plt.subplots(figsize=(10,10))
	for year in range(2014,2022):
		num = pd.read_parquet(f'standardized/{year}_numerator.parquet')
		den0 = pd.read_parquet(f'standardized/{year}_denominator.parquet')

		if icd_code!='R95':
			den = den0.copy()
		else:
			print(f'{year} denominator size total: {den0.shape[0]}')
			if year<2016:
				den = den0.loc[den0['UCOD'].isna()].copy()
			else:
				den = den0.loc[den0['CO_DODYY'].isna()].copy()

			print(f'survivals only size: {den.shape[0]}')
			print(f'difference: {den0.shape[0]-den.shape[0]}')


		############## if data in text format, remove spaces:
		if pd.api.types.is_string_dtype(num['AGED']):
			for idx in num.index:
				num.loc[idx, 'AGED'] = num['AGED'].loc[idx].strip()
				num.loc[idx, 'UCOD'] = num['UCOD'].loc[idx].strip()

		num = num.loc[num['UCOD']==icd_code]

		num['INFAGE']=infage_recode_weeks(num)
		num.dropna(subset=['INFAGE'],inplace=True)	####### don't include any infant younger thn 48 hours
		num = num.copy()

		num.columns = num.columns.str.lower()
		den.columns = den.columns.str.lower()

		s1 = num.shape[0]

		################################### reformat variables
		for vari in variables: num[vari] = num[vari].astype(int)
		for vari in variables[3:len(variables)]: den[vari] = den[vari].astype(int)

		################################### data cleaning
		for variab in cat_vars:
			for idx in num.index:
				if num.loc[idx, variab] == 99:
					num.loc[idx, variab] = np.nan
				if num.loc[idx, variab] == 9:
					num.loc[idx, variab] = np.nan
		num.loc[num.index[num['cig_rec']=='0'], 'cig_rec'] = 'N'
		num.loc[num.index[num['cig_rec']=='1'], 'cig_rec'] = 'Y'
		num.loc[num.index[num['cig_rec']=='U'], 'cig_rec'] = np.nan

		den.loc[den.index[den['cig_rec']=='0'], 'cig_rec'] = 'N'
		den.loc[den.index[den['cig_rec']=='1'], 'cig_rec'] = 'Y'
		den.loc[den.index[den['cig_rec']=='U'], 'cig_rec'] = np.nan

		# num.loc[num.index[num['fage11']==11], 'fage11'] = np.nan
		num.loc[num.index[num['bwtr4']==4], 'bwtr4'] = np.nan
		# num.loc[num.index[num['bmi_r']==9], 'bmi_r'] = np.nan
		num.loc[num.index[num['apgar5']==99], 'apgar5'] = np.nan

		ax.scatter(year, num.shape[0], alpha=0.5, c='k')

		num.dropna(subset=variables,inplace=True)

		################################### reformat variables
		for vari in cat_vars: num[vari] = pd.Categorical(num[vari])
		for vari in cat_vars[0:2]: den[vari] = pd.Categorical(den[vari])

		#########################################################
		s2 = num.shape[0]
		# print(s1,s2)
		ratio = s2/s1
		print(f'{year}\nformer data size: {s1}, filtered size:{s2}, dropped percentage: {np.round(1-ratio,decimals=2)} (subjects with low quality data)\n')
		# print(year,'\n',num[variables].head())
		# print(year,'\n',num[risk_factors].head())
		out0 = pd.concat([out0, num.loc[num.index, variables+cat_vars]],ignore_index=True)
		out1 = pd.concat([out1, den.loc[den.index, variables[3:len(variables)]+cat_vars[0:2]]],ignore_index=True)

		# ax.scatter(year, num.shape[0], c='k')
		ax.scatter(year, s2, c='k')
		ratio1 = np.append(ratio1, ratio)

		print(out0.shape)

	out0.to_parquet(f'standardized/consolidated/{icd_code}_num.parquet')
	if icd_code=='R95':
		out1.to_parquet(f'standardized/consolidated/survivals.parquet')
	else:
		out1.to_parquet(f'standardized/consolidated/{icd_code}_den.parquet')
	fig.savefig(f'output_figures/{icd_code}_samplesize_time.png')
	plt.close()
	return ratio1

def concatenate_numerator(icd_code,variables, cat_vars):
	out0 = pd.DataFrame()
	print(f'\n{icd_code}\n')
	ratio1 = np.array([])
	fig,ax = plt.subplots(figsize=(10,10))
	for year in range(2014,2022):
		num = pd.read_parquet(f'standardized/{year}_numerator.parquet')
		# print(year,num['PRECARE'].value_counts(),'\n')
		############## if data in text format, remove spaces:
		if pd.api.types.is_string_dtype(num['AGED']):
			for idx in num.index:
				num.loc[idx, 'AGED'] = num['AGED'].loc[idx].strip()
				num.loc[idx, 'UCOD'] = num['UCOD'].loc[idx].strip()

		num = num.loc[num['UCOD']==icd_code]

		num['INFAGE']=infage_recode_weeks(num)
		num.dropna(subset=['INFAGE'],inplace=True)	####### don't include any infant younger thn 48 hours
		num = num.copy()

		num.columns = num.columns.str.lower()

		s1 = num.shape[0]

		################################### reformat variables
		for vari in variables: num.loc[:,vari] = num[vari].astype(int)

		################################### data cleaning
		for variab in cat_vars:
			for idx in num.index:
				if num.loc[idx, variab] == 99:
					num.loc[idx, variab] = np.nan
				if num.loc[idx, variab] == 9:
					num.loc[idx, variab] = np.nan
		num.loc[num.index[num['cig_rec']=='0'], 'cig_rec'] = 'N'
		num.loc[num.index[num['cig_rec']=='1'], 'cig_rec'] = 'Y'
		num.loc[num.index[num['cig_rec']=='U'], 'cig_rec'] = np.nan

		num.loc[num.index[num['bwtr4']==4], 'bwtr4'] = np.nan
		num.loc[num.index[num['apgar5']==99], 'apgar5'] = np.nan

		ax.scatter(year, num.shape[0], alpha=0.5, c='k')

		num.dropna(subset=variables,inplace=True)

		################################### reformat variables
		for vari in cat_vars: num.loc[:,vari] = pd.Categorical(num[vari])
		#########################################################
		s2 = num.shape[0]
		ratio = s2/s1
		print(f'{year}\nformer data size: {s1}, filtered size:{s2}, dropped percentage: {np.round(1-ratio,decimals=2)} (subjects with low quality data)\n\n#########################\n')
		out0 = pd.concat([out0, num.loc[num.index, variables+cat_vars]],ignore_index=True)

		ax.scatter(year, s2, c='k')
		ratio1 = np.append(ratio1, ratio)

		print(out0.shape)

	out0.to_parquet(f'standardized/consolidated/{icd_code}_num.parquet')

	fig.savefig(f'output_figures/{icd_code}_samplesize_time.png')
	plt.close()
	return ratio1

def concatenate_denominator(variables, cat_vars):
	out1 = pd.DataFrame()
	print('\nDenominator (all US live births)',)
	for year in range(2014,2022):
		den = pd.read_parquet(f'standardized/{year}_denominator.parquet')

		############## if data in text format, remove spaces:
		den.columns = den.columns.str.lower()
		################################### reformat variables
		print(year, den.shape[0])
		for vari in variables: den.loc[:, vari] = den[vari].astype(int)

		################################### data cleaning
		den.loc[den.index[den['cig_rec']=='0'], 'cig_rec'] = 'N'
		den.loc[den.index[den['cig_rec']=='1'], 'cig_rec'] = 'Y'
		den.loc[den.index[den['cig_rec']=='U'], 'cig_rec'] = np.nan

		# den.dropna(subset=['cig_rec'],inplace=True)

		################################### reformat variables
		for vari in cat_vars: den.loc[:,vari] = pd.Categorical(den[vari])

		#########################################################

		out1 = pd.concat([out1, den.loc[den.index, variables+cat_vars]],ignore_index=True)

	out1.to_parquet(f'standardized/consolidated/denominator.parquet',compression='snappy',engine='pyarrow')

def concatenate_survivals(variables, cat_vars):
	out1 = pd.DataFrame()

	print('\nSurvivals (US live births not associated with variables: ICD code nor death year)',)

	for year in range(2014,2022):
		den0 = pd.read_parquet(f'standardized/{year}_denominator.parquet')

		################## filter out deads
		print(f'{year} denominator size total: {den0.shape[0]}')
		if year<2016:
			den0.loc[den0['UCOD']=='1.00', 'UCOD']= np.nan
			den = den0.loc[den0['UCOD'].isna()].copy()
		elif year==2021:
			den0.loc[den0['CO_DODYY']=='    ', 'CO_DODYY'] = np.nan
			den = den0.loc[den0['CO_DODYY'].isna().copy()]
		else:
			den = den0.loc[den0['CO_DODYY'].isna()].copy()

		den=den.copy()
		print(f'survivals only size: {den.shape[0]}')
		print(f'difference: {den0.shape[0]-den.shape[0]}')

		############## if data in text format, remove spaces:
		den.columns = den.columns.str.lower()
		################################### reformat variables
		for vari in variables: den.loc[:, vari] = den[vari].astype(int)

		################################### data cleaning
		den.loc[den['cig_rec']=='0', 'cig_rec'] = 'N'
		den.loc[den['cig_rec']=='1', 'cig_rec'] = 'Y'
		den.loc[den['cig_rec']=='U', 'cig_rec'] = np.nan

		################################### reformat variables
		for vari in cat_vars: den.loc[:,vari] = pd.Categorical(den[vari])

		#########################################################

		out1 = pd.concat([out1, den.loc[den.index, variables+cat_vars]],ignore_index=True)

	out1.to_parquet(f'standardized/consolidated/survivals.parquet',compression='snappy',engine='pyarrow')

def concatenate_dead(variables, cat_vars):
	out1 = pd.DataFrame()

	print('\nDead (US live births associated with any ICD code or death year)',)

	for year in range(2014,2022):
		num = pd.read_parquet(f'standardized/{year}_numerator.parquet')

		################## filter out deads
		print(f'{year} numerator size total: {num.shape[0]}')

		num['INFAGE']=infage_recode_weeks(num)
		num.dropna(subset=['INFAGE'],inplace=True)	####### don't include any infant younger thn 48 hours
		num = num.copy()

		############## if data in text format, remove spaces:
		num.columns = num.columns.str.lower()
		################################### reformat variables
		for vari in variables: num.loc[:,vari] = num[vari].astype(int)

		################################### data cleaning
		num.loc[num['cig_rec']=='0', 'cig_rec'] = 'N'
		num.loc[num['cig_rec']=='1', 'cig_rec'] = 'Y'
		num.loc[num['cig_rec']=='U', 'cig_rec'] = np.nan

		num.dropna(subset=['cig_rec'],inplace=True)
		################################### reformat variables
		for vari in cat_vars: num.loc[:,vari] = pd.Categorical(num[vari])

		#########################################################

		out1 = pd.concat([out1, num.loc[:, variables+cat_vars]],ignore_index=True)
	# print(out1['infage'].value_counts().sort_index())
	out1.to_parquet(f'standardized/consolidated/dead.parquet')
#### Basic tasks:
#### iterate through the years of interest.
#### collect variables on interest and perform basic data cleaning on the fly.
#### concatenate all years into a single file. Save data

def main():
	###### 2 days and older:
	##################### icd code, list of numeric variables, list of categorical variables, infant age category. infant for older than 2 days
	# list1 = ['infage', 'mager14','dob_mm', 'dob_wk', 'dob_yy','bwtr4','bmi_r']
	list0 = ['infage', 'mager14','dob_mm', 'dob_wk', 'dob_yy','bwtr4', 'apgar5', 'precare', 'race']

	den_list = ['apgar10', 'apgar10r', 'apgar5', 'apgar5r', 'attend', 'bfacil', 'bfacil3', 'bmi_r', 'brthwgt', 'bwtr14', 'bwtr4', 'cig0_r', 'cig1_r', 'cig2_r', 'cig3_r', 'cig_0', 'cig_1', 'cig_2', 'cig_3', 'combgest', 'dlmp_mm', 'dlmp_yy', 'dmeth_rec', 'dob_mm', 'dob_tt', 'dob_wk', 'dob_yy', 'dplural', 'dwgt_r', 'fage11', 'fagecomb', 'feduc', 'fhisp_r', 'frace15', 'frace31', 'frace6', 'fracehisp', 'illb_r', 'illb_r11', 'ilp_r', 'ilp_r11', 'lbo_rec', 'mager', 'mager14', 'mager9', 'mbstate_rec', 'meduc', 'me_pres', 'me_rout', 'mhisp_r', 'mhtr', 'mrace15', 'mrace31', 'mrace6', 'mracehisp', 'no_abnorm', 'no_congen', 'no_infec', 'no_lbrdlv', 'no_mmorb', 'no_risks', 'oegest_comb', 'rdmeth_rec', 'restatus', 'wtgain', 'wtgain_rec','precare']

	num_list = ['infage', 'aged', 'apgar10', 'apgar10r', 'apgar5', 'apgar5r', 'attend', 'bfacil', 'bfacil3', 'bmi_r', 'brthwgt', 'bwtr14', 'bwtr4', 'cig0_r', 'cig1_r', 'cig2_r', 'cig3_r', 'cig_0', 'cig_1', 'cig_2', 'cig_3', 'combgest', 'dlmp_mm', 'dlmp_yy', 'dmeth_rec', 'dob_mm', 'dob_tt', 'dob_wk', 'dob_yy', 'dod_mm', 'dod_yy', 'dplural', 'dweekday', 'dwgt_r', 'eanum', 'fage11', 'fagecomb', 'feduc', 'fhisp_r', 'flgnd', 'frace15', 'frace31', 'frace6', 'fracehisp', 'hospd', 'illb_r', 'illb_r11', 'ilp_r', 'ilp_r11', 'lbo_rec', 'mager', 'mager14', 'mager9', 'mbstate_rec', 'meduc', 'me_pres', 'me_rout', 'mhisp_r', 'mhtr', 'mrace15', 'mrace31', 'mrace6', 'mracehisp', 'no_abnorm', 'no_congen', 'no_infec', 'no_lbrdlv', 'no_mmorb', 'no_risks', 'oegest_comb', 'ranum', 'rdmeth_rec', 'restatus', 'ucodr130', 'wtgain', 'wtgain_rec','precare']

	#### to do:
	#### create one list for numerator - done
	#### create another list for denominator - done
	#### list2 has all categorical variables; only the first item missing in denominator
	### list0 has all categorical variables; the first 3 items are missing in denominator

	### make sure items are read as int64 in num and den lists
	#### figure out why live 2014 and 2015 only have 1

	Klist = ['ucod', 'sex', 'cig_rec', 'ca_anen', 'ca_mnsb', 'ca_cchd', 'ca_cdh', 'ca_omph', 'ca_gast', 'ab_aven1', 'ab_aven6', 'ab_nicu', 'ab_surf', 'ab_anti', 'ab_seiz']
	olist = ['autopsy', 'dispo', 'manner', 'bfed', 'bwtimp', 'ilive', 'imp_plur', 'ip_chlam', 'ip_gon', 'ip_hepb', 'ip_hepc', 'ip_syph', 'itran', 'ld_anes', 'ld_antb', 'ld_augm', 'ld_chor', 'ld_indl', 'ld_ster', 'mar_p', 'me_trial', 'mm_aicu', 'mm_mtr', 'mm_plac', 'mm_rupt', 'mm_uhyst', 'mtran', 'ob_fail', 'ob_succ', 'setorder_r',
	'wic']

	r1 = concatenate_numerator('R95', num_list, Klist+olist)
	# print(r1)
	r2 = concatenate_numerator('W75', num_list, Klist+olist)
	r3 = concatenate_numerator('R99', num_list, Klist+olist)
	# concatenate_denominator(den_list, Klist[1:len(Klist)]+olist[3:len(olist)])
	# concatenate_survivals(den_list, Klist[1:len(Klist)]+olist[3:len(olist)])
	concatenate_dead(num_list, Klist+olist)

	# r4 = concatenate_years('P072',num_list, list2)
	# r5 = concatenate_years('P073',num_list, list2)
	# r6 = concatenate_years('P011',listR99, list2)
	# r7 = concatenate_years('Q913',listR99, list2)
	# r8 = concatenate_years('Q249',listR99, list2)

	# average_kept1 = np.array([r1.mean(), r2.mean(), r3.mean()])#, r4.mean(), r5.mean(), r6.mean(), r7.mean(), r8.mean()]).mean()
	# print(f'\naverage kept: {average_kept1}')


	# os.makedirs('output_data/concatenate_logs',exist_ok=True)
	# np.save('output_data/concatenate_logs/infant.npy',average_kept1)

if __name__=="__main__":
	main()