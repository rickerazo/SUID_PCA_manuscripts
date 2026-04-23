import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

pd.set_option('display.max_columns', None)
#####################################
def load_and_reformat(icd_code,fraction, list1):
	df = pd.read_parquet(f'standardized/consolidated/{icd_code}_{fraction}.parquet')

	df.loc[df['mager14']==1,'mother_age'] = 'U15'
	df.loc[df['mager14']==3,'mother_age'] = '15'
	df.loc[df['mager14']==4,'mother_age'] = '16'
	df.loc[df['mager14']==5,'mother_age'] = '17'
	df.loc[df['mager14']==6,'mother_age'] = '18'
	df.loc[df['mager14']==7,'mother_age'] = '19'
	df.loc[df['mager14']==8,'mother_age'] = '20-24'
	df.loc[df['mager14']==9,'mother_age'] = '25-29'
	df.loc[df['mager14']==10,'mother_age'] = '30-34'
	df.loc[df['mager14']==11,'mother_age'] = '35-39'
	df.loc[df['mager14']==12,'mother_age'] = '40-44'
	df.loc[df['mager14']==13,'mother_age'] = '45-49'
	df.loc[df['mager14']==14,'mother_age'] = '50-54'

	df.loc[df['dob_mm']==1,'birth_month'] = 'Jan'
	df.loc[df['dob_mm']==2,'birth_month'] = 'Feb'
	df.loc[df['dob_mm']==3,'birth_month'] = 'Mar'
	df.loc[df['dob_mm']==4,'birth_month'] = 'Apr'
	df.loc[df['dob_mm']==5,'birth_month'] = 'May'
	df.loc[df['dob_mm']==6,'birth_month'] = 'Jun'
	df.loc[df['dob_mm']==7,'birth_month'] = 'Jul'
	df.loc[df['dob_mm']==8,'birth_month'] = 'Aug'
	df.loc[df['dob_mm']==9,'birth_month'] = 'Sep'
	df.loc[df['dob_mm']==10,'birth_month'] = 'Oct'
	df.loc[df['dob_mm']==11,'birth_month'] = 'Nov'
	df.loc[df['dob_mm']==12,'birth_month'] = 'Dec'

	df.loc[df['dob_wk']==1,'birth_day'] = '1'
	df.loc[df['dob_wk']==2,'birth_day'] = '2'
	df.loc[df['dob_wk']==3,'birth_day'] = '3'
	df.loc[df['dob_wk']==4,'birth_day'] = '4'
	df.loc[df['dob_wk']==5,'birth_day'] = '5'
	df.loc[df['dob_wk']==6,'birth_day'] = '6'
	df.loc[df['dob_wk']==7,'birth_day'] = '7'

	# df.loc[df['fage11']==1,'father_age'] = 'U15'
	# df.loc[df['fage11']==2,'father_age'] = '15-19'
	# df.loc[df['fage11']==3,'father_age'] = '20-24'
	# df.loc[df['fage11']==4,'father_age'] = '25-29'
	# df.loc[df['fage11']==5,'father_age'] = '30-34'
	# df.loc[df['fage11']==6,'father_age'] = '35-39'
	# df.loc[df['fage11']==7,'father_age'] = '40-44'
	# df.loc[df['fage11']==8,'father_age'] = '45-49'
	# df.loc[df['fage11']==9,'father_age'] = '50-54'
	# df.loc[df['fage11']==10,'father_age'] = '55-98'

	df.loc[df['bwtr4']==1,'bw'] = '0227-1499'
	df.loc[df['bwtr4']==2,'bw'] = '1500-2499'
	df.loc[df['bwtr4']==3,'bw'] = '2500-8165'

	# df.loc[df['bmi_r']==1,'bmi'] = 'underweight'
	# df.loc[df['bmi_r']==2,'bmi'] = 'normal'
	# df.loc[df['bmi_r']==3,'bmi'] = 'overweight'
	# df.loc[df['bmi_r']==4,'bmi'] = 'obesityI'
	# df.loc[df['bmi_r']==5,'bmi'] = 'obesityII'
	# df.loc[df['bmi_r']==6,'bmi'] = 'obesityIII'

	df.loc[df['cig_rec']=='2', 'cig_rec'] = 'N'
	df.loc[df['cig_rec']=='7', 'cig_rec'] = 'N'
	# if fraction =='num':
		# df.loc[df['manner']==1,'way'] = 'accident'
		# df.loc[df['manner']==2,'way'] = 'suicide'
		# df.loc[df['manner']==3,'way'] = 'homicide'
		# df.loc[df['manner']==4,'way'] = 'pending'
		# df.loc[df['manner']==5,'way'] = 'undetermind'
		# df.loc[df['manner']==6,'way'] = 'self-inflictd'
		# df.loc[df['manner']==7,'way'] = 'natural'

	# 	variables = ['infage', 'mother_age', 'sex', 'birth_month', 'birth_day', 'father_age','cig_rec', 'manner','bw','no_abnorm','bmi', 'apgar5','rf_pdiab', 'rf_gdiab', 'rf_phype', 'rf_ghype','rf_ehype', 'rf_ppb']
	# else:
		# variables = ['mother_age', 'sex', 'birth_month', 'birth_day', 'father_age','cig_rec', 'bw','no_abnorm','bmi', 'apgar5','rf_pdiab', 'rf_gdiab', 'rf_phype', 'rf_ghype','rf_ehype', 'rf_ppb']
					# ['infage', 'mager14','dob_mm', 'dob_wk','bwtr4','bmi_r','sex', 'cig_rec']
	# variables = ['infage', 'mager14','dob_mm', 'dob_wk','bwtr4','bmi_r', 'sex', 'cig_rec']
	if fraction =='num':
		# variables = ['infage','mother_age', 'birth_month', 'birth_day','bw','bmi','sex','cig_rec','dob_yy']	
		variables = ['infage','mother_age', 'birth_month', 'birth_day','bw','sex','cig_rec','dob_yy','ca_anen', 'ca_mnsb', 'ca_cchd', 'ca_cdh', 'ca_omph', 'ca_gast', 'ab_aven1', 'ab_aven6', 'ab_nicu', 'ab_surf', 'ab_anti', 'ab_seiz', 'apgar5']
		variables = variables+['dplural','dmeth_rec','cig_3','attend','cig_1','mager14','cigs','wtgain_rec','cig_2','apgar5r','wtgain','dlmp_yy','combgest','bwtr14','restatus','dlmp_mm','bfacil3']
	else: 
		variables = ['mother_age', 'birth_month', 'birth_day','bw','sex','cig_rec','dob_yy','apgar5']
		variables = variables+['dplural','dmeth_rec','cig_3','attend','cig_1','mager14','cigs','wtgain_rec','cig_2','apgar5r','wtgain','dlmp_yy','combgest','bwtr14','restatus','dlmp_mm','bfacil3']
		# variables = ['mother_age', 'birth_month', 'birth_day','bw','bmi','sex','cig_rec','dob_yy']	

	# else:
	# 	variables = ['mother_age', 'sex', 'birth_month', 'birth_day', 'father_age','cig_rec', 'bw','no_abnorm','bmi', 'apgar5','rf_pdiab', 'rf_gdiab', 'rf_phype', 'rf_ghype','rf_ehype', 'rf_ppb']



	out = pd.DataFrame()
	out = df[variables]
	return out

###############################################################################

def compute_DRs(icd_code, num, den):
	print(icd_code)
	###############################################################################
	###############################################################################
	### death rate as a function of infant age
	smoker = 'N'
	smoker_M_death_rate_infage = np.zeros(53)
	smoker_F_death_rate_infage = np.zeros(53)
	nonsmk_M_death_rate_infage = np.zeros(53)
	nonsmk_F_death_rate_infage = np.zeros(53)

	smoker_death_rate_infage = np.zeros(53)
	nonsmk_death_rate_infage = np.zeros(53)
	death_rate_infage = np.zeros(53)
	i=0
	for age in range(0,53):
		cohort = num.loc[num['infage']==age]
		cohort_smk = cohort.loc[cohort['cig_rec']=='Y']
		cohort_nsm = cohort.loc[cohort['cig_rec']=='N']
		smoker_death_rate_infage[i] = cohort_smk.shape[0]/den.loc[den['cig_rec']=='Y'].shape[0]
		nonsmk_death_rate_infage[i] = cohort_nsm.shape[0]/den.loc[den['cig_rec']=='N'].shape[0]

		death_rate_infage[i] = cohort.shape[0]/den.shape[0]

		idx_M = set(den.index[den['sex']=='M'])
		idx_F = set(den.index[den['sex']=='F'])
		idx_smoke = set(den.index[den['cig_rec']=='Y'])
		idx_nosmk = set(den.index[den['cig_rec']=='N'])

		#### smoker male DR
		if (cohort_smk['sex']=='M').any():
			smoker_M_death_rate_infage[i] = cohort_smk['sex'].value_counts().loc['M']/ den.loc[list(idx_M.intersection(idx_smoke))].shape[0]
		#### smoker female DR
		if (cohort_smk['sex']=='F').any():
			smoker_F_death_rate_infage[i] = cohort_smk['sex'].value_counts().loc['F']/ den.loc[list(idx_F.intersection(idx_smoke))].shape[0]

		#### nonsmoker male DR
		if (cohort_nsm['sex']=='M').any():
			nonsmk_M_death_rate_infage[i] = cohort_nsm['sex'].value_counts().loc['M']/ den.loc[list(idx_M.intersection(idx_nosmk))].shape[0]
		#### nonsmoker female DR
		if (cohort_nsm['sex']=='F').any():
			nonsmk_F_death_rate_infage[i] = cohort_nsm['sex'].value_counts().loc['F']/ den.loc[list(idx_F.intersection(idx_nosmk))].shape[0]
		i+=1
		print(f'processing weeks {i}')
	### death rate as a function of season: birth month
	smoker_death_rate_month = np.zeros(12)
	smoker_M_death_rate_month = np.zeros(12)
	smoker_F_death_rate_month = np.zeros(12)

	nonsmk_death_rate_month = np.zeros(12)
	nonsmk_M_death_rate_month = np.zeros(12)
	nonsmk_F_death_rate_month = np.zeros(12)

	season_death_rate = np.zeros(12)
	i=0
	year = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

	for month in year:
		season_death_rate[i] = num.loc[num['birth_month']==month].shape[0]/den.loc[den['birth_month']==month].shape[0]
		
		cohort = num.loc[num['birth_month']==month]
		cohort_smk = cohort.loc[cohort['cig_rec']=='Y']
		cohort_nsm = cohort.loc[cohort['cig_rec']=='N']
		smoker_death_rate_month[i] = cohort_smk.shape[0]/den.loc[den['cig_rec']=='Y'].shape[0]
		nonsmk_death_rate_month[i] = cohort_nsm.shape[0]/den.loc[den['cig_rec']=='N'].shape[0]

		idx_M = set(den.index[den['sex']=='M'])
		idx_F = set(den.index[den['sex']=='F'])
		idx_smoke = set(den.index[den['cig_rec']=='Y'])
		idx_nosmk = set(den.index[den['cig_rec']=='N'])


		#### smoker male DR
		if (cohort_smk['sex']=='M').any():
			smoker_M_death_rate_month[i] = cohort_smk['sex'].value_counts().loc['M']/ den.loc[list(idx_M.intersection(idx_smoke))].shape[0]
		#### smoker female DR
		if (cohort_smk['sex']=='F').any():
			smoker_F_death_rate_month[i] = cohort_smk['sex'].value_counts().loc['F']/ den.loc[list(idx_F.intersection(idx_smoke))].shape[0]

		#### nonsmoker male DR
		if (cohort_nsm['sex']=='M').any():
			nonsmk_M_death_rate_month[i] = cohort_nsm['sex'].value_counts().loc['M']/ den.loc[list(idx_M.intersection(idx_nosmk))].shape[0]
		#### nonsmoker female DR
		if (cohort_nsm['sex']=='F').any():
			nonsmk_F_death_rate_month[i] = cohort_nsm['sex'].value_counts().loc['F']/ den.loc[list(idx_F.intersection(idx_nosmk))].shape[0]

		i+=1

		print(f'processing months {i}')
	#############################################
	num['age_DR'] = 0.
	num['smk_age_DR'] = 0.
	num['smk_age_sex_DR'] = 0.
	for i in range(0,53):
		num.loc[num['infage']==i, 'age_DR'] = death_rate_infage[i]

		p1 = set(num.index[num['infage']==i])

		p2 = set(num.index[num['cig_rec']=='Y'])
		p3 = set(num.index[num['cig_rec']=='N'])


		##### smoker DR, all sexes
		smoker_idx = p1.intersection(p2)
		nonsmk_idx = p1.intersection(p3)
		num.loc[list(smoker_idx), 'smk_age_DR'] = smoker_death_rate_infage[i]
		num.loc[list(nonsmk_idx), 'smk_age_DR'] = nonsmk_death_rate_infage[i]

		################################
		p4 = set(num.index[num['sex']=='M'])
		#### smoker DR, males
		smoker_M_idx = smoker_idx.intersection(p4)
		num.loc[list(smoker_M_idx), 'smk_age_sex_DR'] = smoker_M_death_rate_infage[i]
		#### nonsmoker DR, males
		nonsmk_M_idx = nonsmk_idx.intersection(p4)
		num.loc[list(nonsmk_M_idx), 'smk_age_sex_DR'] = nonsmk_M_death_rate_infage[i]

		
		################################
		p5 = set(num.index[num['sex']=='F'])
		#### smoker DR, females
		smoker_F_idx = smoker_idx.intersection(p5)
		num.loc[list(smoker_F_idx), 'smk_age_sex_DR'] = smoker_F_death_rate_infage[i]
		#### nonsmoker DR, females
		nonsmk_F_idx = nonsmk_idx.intersection(p5)
		num.loc[list(nonsmk_F_idx), 'smk_age_sex_DR'] = nonsmk_F_death_rate_infage[i]

		print(f'storing weeks {i}')

	num['seasonal_DR'] = 0.
	num['seasonal_smk_DR'] = 0.
	num['seasonal_smk_M_DR'] = 0.
	num['seasonal_smk_F_DR'] = 0.
	num['seasonal_nsm_DR'] = 0.
	num['seasonal_nsm_M_DR'] = 0.
	num['seasonal_nsm_F_DR'] = 0.
	i=0
	for month in year:
		num.loc[num['birth_month']==month, 'seasonal_DR'] = season_death_rate[i]

		nonsmks = set(num.index[num['cig_rec']=='N'])
		smokers = set(num.index[num['cig_rec']=='Y'])
		mo = set(num.index[num['birth_month']==month])
		ma = set(num.index[num['sex']=='M'])
		fe = set(num.index[num['sex']=='F'])

		### all smokers
		num.loc[list(smokers.intersection(mo)), 'seasonal_smk_DR'] = smoker_death_rate_month[i]
		smok_cohort = smokers.intersection(mo)

		### male smokers
		num.loc[list(smok_cohort.intersection(ma)), 'seasonal_smk_M_DR'] = smoker_M_death_rate_month[i]
		### female smokers
		num.loc[list(smok_cohort.intersection(fe)), 'seasonal_smk_F_DR'] = smoker_F_death_rate_month[i]


		#### all nonsmokers
		num.loc[list(nonsmks.intersection(mo)), 'seasonal_nsm_DR'] = nonsmk_death_rate_month[i]
		nosk_cohort = nonsmks.intersection(mo)

		### male nonsmokers
		num.loc[list(nosk_cohort.intersection(ma)), 'seasonal_nsm_M_DR'] = nonsmk_M_death_rate_month[i]
		### female nonsmokers
		num.loc[list(nosk_cohort.intersection(fe)), 'seasonal_nsm_F_DR'] = nonsmk_F_death_rate_month[i]
		i+=1

		print(f'storing months {i}')
	print(num.shape,'\n',num.head())


	num.to_parquet(f'standardized/consolidated/{icd_code}_death_rates.parquet')

###############################################################################	INFANT:
###############################################################################	infant age
############################################################################### older than 2 days
# variables1 = ['infage', 'mager14','dob_mm', 'dob_wk','bwtr4','bmi_r','sex', 'cig_rec']
variables1 = ['infage', 'mager14','dob_mm', 'dob_wk','bwtr4','sex', 'cig_rec']
icd_code = 'R95'
###### Load numerator and denominator

num0 = load_and_reformat(icd_code,'num', variables1)
den0 = load_and_reformat(icd_code,'den', variables1[1:len(variables1)])

num0.to_parquet(f'standardized/consolidated/{icd_code}_numerator.parquet')
den0.to_parquet(f'standardized/consolidated/{icd_code}_denominator.parquet')
# compute_DRs(icd_code, num0, den0)

###############################################################################
icd_code = 'W75'
###### Load numerator and denominator
num1 = load_and_reformat(icd_code,'num', variables1)
den1 = load_and_reformat(icd_code,'den', variables1[1:len(variables1)])

# compute_DRs(icd_code, num1, den1)
num1.to_parquet(f'standardized/consolidated/{icd_code}_numerator.parquet')
den1.to_parquet(f'standardized/consolidated/{icd_code}_denominator.parquet')

###############################################################################
icd_code = 'R99'
###### Load numerator and denominator
num2 = load_and_reformat(icd_code,'num', variables1)
den2 = load_and_reformat(icd_code,'den', variables1[1:len(variables1)])

# compute_DRs(icd_code, num2, den2)

num2.to_parquet(f'standardized/consolidated/{icd_code}_numerator.parquet')
den2.to_parquet(f'standardized/consolidated/{icd_code}_denominator.parquet')

###############################################################################
icd_code = 'P072'
###### Load numerator and denominator
num3 = load_and_reformat(icd_code,'num', variables1)
den3 = load_and_reformat(icd_code,'den', variables1[1:len(variables1)])

# compute_DRs(icd_code, num3, den3)
num3.to_parquet(f'standardized/consolidated/{icd_code}_numerator.parquet')
den3.to_parquet(f'standardized/consolidated/{icd_code}_denominator.parquet')

###############################################################################
icd_code = 'P073'
###### Load numerator and denominator
num4 = load_and_reformat(icd_code,'num', variables1)
den4 = load_and_reformat(icd_code,'den', variables1[1:len(variables1)])

# compute_DRs(icd_code, num4, den4)
num4.to_parquet(f'standardized/consolidated/{icd_code}_numerator.parquet')
den4.to_parquet(f'standardized/consolidated/{icd_code}_denominator.parquet')

###############################################################################
icd_code = 'P011'
###### Load numerator and denominator
num5 = load_and_reformat(icd_code,'num', variables1)
den5 = load_and_reformat(icd_code,'den', variables1[1:len(variables1)])

# compute_DRs(icd_code, num5, den5)
num5.to_parquet(f'standardized/consolidated/{icd_code}_numerator.parquet')
den5.to_parquet(f'standardized/consolidated/{icd_code}_denominator.parquet')

###############################################################################
icd_code = 'Q913'
###### Load numerator and denominator
num6 = load_and_reformat(icd_code,'num', variables1)
den6 = load_and_reformat(icd_code,'den', variables1[1:len(variables1)])

# compute_DRs(icd_code, num6, den6)
num6.to_parquet(f'standardized/consolidated/{icd_code}_numerator.parquet')
den6.to_parquet(f'standardized/consolidated/{icd_code}_denominator.parquet')

###############################################################################
icd_code = 'Q249'
###### Load numerator and denominator
num7 = load_and_reformat(icd_code,'num', variables1)
den7 = load_and_reformat(icd_code,'den', variables1[1:len(variables1)])

# compute_DRs(icd_code, num7, den7)
num7.to_parquet(f'standardized/consolidated/{icd_code}_numerator.parquet')
den7.to_parquet(f'standardized/consolidated/{icd_code}_denominator.parquet')
