'''
script to compute risk exposure for the population offset
for NB models and any other downstream regression models that take in risk exposure as a variable to correct for representation sample vs population

PRAMS data from 2012 to 2022

WHAT THIS FIXES vs. risk_exposure.py
-------------------------------------
compute_preterm() set df['preterm']=0 for everyone, then conditionally wrote
a *different* column, df['term'], to 1 for combgest<37 weeks, and returned
the untouched all-zero 'preterm' column. The caller did
`den['term'] = compute_preterm(den)`, so 'term' ended up all zeros instead
of holding the intended preterm flag.

Consequence: compute_exposure(den, 'term', 0, 1) builds
{0: risk_.loc[0], 1: risk_.loc[1]} from that all-zero column. Category 1
never appears, so risk_.loc[1] raises KeyError — the script cannot reach the
line that writes term_exposure.pkl in its current form. That pickle holds
the prematurity offset every negative binomial model in the pipeline
multiplies in, so this blocks (or, if some other version of this file
produced the pickles currently on disk, silently disagrees with) every
model's prematurity term.

Fix: compute the flag and return the SAME column, under a name that means
what it says (preterm=1 for combgest<37, matching the manuscript's own
"preterm birth, defined as less than 37 completed gestational weeks").
Also dropped the two dead module-level variables (option1/option2) that
looked like they configured compute_exposure()'s call but were never
actually used — the real call below passes literal 'N'/'Y' strings instead,
which was silently correct but confusing.
'''

# risk_exposure.py
import pickle
import os
os.makedirs('output_data/risk_exposure', exist_ok=True)
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

den = pd.read_parquet('/data/hps/assoc/private/ramirezlab_sids/ricardo/data_science/infant_mortality/standardized/consolidated/denominator.parquet')
df_ = pd.read_parquet('/data/hps/assoc/private/ramirezlab_sids/ricardo/data_science/controls/output_data/PRAMS/prams.parquet')


def compute_exposure(df, var, option1, option2):
    risk_ = df[var].value_counts() / df[var].value_counts().sum()
    exposure = {0: risk_.loc[option1], 1: risk_.loc[option2]}
    return exposure


def compute_preterm(df):
    '''
    preterm=1 if gestational age < 37 completed weeks, else 0.
    Excludes rows with an unknown/not-stated combgest (NCHS sentinel >=99)
    before computing the proportion, matching the case-side cleaning in
    load_case_control() step 2. Left in, those rows would silently count as
    "term" (99 is not < 37), understating the true preterm rate.
    '''
    known = df.loc[df['combgest'] < 99].copy()
    known['preterm'] = (known['combgest'] < 37).astype(int)
    df.loc[known.index, 'preterm'] = known['preterm']
    return df.loc[known.index, 'preterm']


def compute_bed_share(df_):
    df_['bedshare'] = 1
    df_.loc[df_['sleepown_raw'] == 1, 'bedshare'] = 0
    df_.loc[df_['sleepown_raw'] == 2, 'bedshare'] = 0

    '''
    FROM PRAMS documentation:
    39	SLEEPOWN_RAW	Sleep -- baby alone	FREQ5F	.B=BLANK/DK
                                                    .S=SKIP
                                                    1=ALWAYS
                                                    2=OFTEN/ALMOST ALWAYS
                                                    3=SOMETIMES
                                                    4=RARELY
                                                    5=NEVER

    NOTE: blank/DK/skip responses fall through to the bedshare=1 default
    (i.e. non-response is currently treated as bed-sharing, not as missing).
    Not touched by this fix — flagging it here since it silently shapes
    bedshare_exposure if/when that's used for the NCFRP sub-analysis.
    '''
    return df_['bedshare']


den['preterm'] = compute_preterm(den)
dic1 = compute_exposure(den, 'cig_rec', "N", "Y")
dic2 = compute_exposure(den, 'sex', "F", "M")
dic3 = compute_exposure(den, 'preterm', 0, 1)

df_['bedshare'] = compute_bed_share(df_)
dic4 = compute_exposure(df_, 'bedshare', 0, 1)

smoke_exposure = pd.DataFrame([dic1])
sex_exposure = pd.DataFrame([dic2])
term_exposure = pd.DataFrame([dic3])
bedshare_exposure = pd.DataFrame([dic4])

print('smoke_exposure:', dic1)
print('sex_exposure:', dic2)
print('term_exposure (preterm flag, <37wk):', dic3)
print('bedshare_exposure:', dic4)

smoke_exposure.to_csv('output_data/risk_exposure/smoke_exposure.csv')
sex_exposure.to_csv('output_data/risk_exposure/sex_exposure.csv')
term_exposure.to_csv('output_data/risk_exposure/term_exposure.csv')
bedshare_exposure.to_csv('output_data/risk_exposure/bedshare_exposure.csv')

with open('output_data/risk_exposure/smoke_exposure.pkl', 'wb') as f:
    pickle.dump(dic1, f)
with open('output_data/risk_exposure/sex_exposure.pkl', 'wb') as f:
    pickle.dump(dic2, f)
with open('output_data/risk_exposure/term_exposure.pkl', 'wb') as f:
    pickle.dump(dic3, f)
with open('output_data/risk_exposure/bedshare_exposure.pkl', 'wb') as f:
    pickle.dump(dic4, f)


########

d = den.copy()
d["smoke01"] = (d["cig_rec"] == "Y").astype(int)
d["preterm"] = (d["combgest"] < 37).astype(int)  # or <38 if that's your definition

tab = pd.crosstab(d["smoke01"], d["preterm"])
chi2, p, dof, expected = chi2_contingency(tab)

# effect size: phi for 2x2
n = tab.to_numpy().sum()
phi = np.sqrt(chi2 / n)

# odds ratio (with a tiny continuity correction to avoid divide-by-zero)
a, b, c, e = tab.to_numpy().ravel()  # rows: smoke 0/1, cols: preterm 0/1
cc = 0.5
OR = ((e + cc) * (a + cc)) / ((b + cc) * (c + cc))

print(tab)
print(f"smoking x preterm chi2 p-value = {p:.3g}")
print(f"phi = {phi:.4f}")
print(f"odds ratio (smoking -> preterm) ~ {OR:.3f}")

########
# Added: smoking x infant sex, same pattern as above. Checks the reviewer's
# claim that maternal smoking is associated with giving birth to a female
# infant, directly on this dataset rather than asserting it from memory.

d["female01"] = (d["sex"] == "F").astype(int)

tab2 = pd.crosstab(d["smoke01"], d["female01"])
chi2_2, p_2, dof_2, expected_2 = chi2_contingency(tab2)

n2 = tab2.to_numpy().sum()
phi_2 = np.sqrt(chi2_2 / n2)

a2, b2, c2, e2 = tab2.to_numpy().ravel()  # rows: smoke 0/1, cols: female 0/1
OR_2 = ((e2 + cc) * (a2 + cc)) / ((b2 + cc) * (c2 + cc))

print(tab2)
print(f"smoking x female-sex chi2 p-value = {p_2:.3g}")
print(f"phi = {phi_2:.4f}")
print(f"odds ratio (smoking -> female infant) ~ {OR_2:.3f}")
