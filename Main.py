import csv
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.multicomp import pairwise_tukeyhsd


csv_dir = 'Group_11_RunTable.csv'
data = []
data_reader = csv.reader(open(csv_dir))
for line in data_reader:
    data.append(line)
data = pd.DataFrame(data)
data.dropna()
dx = data.iloc[1:, 2].astype('float')
dy = data.iloc[1:, 3].astype('float')
needle = data.iloc[1:, 5]
mass = data.iloc[1:, 6]

dataframe = pd.DataFrame({'Needle': needle.values.tolist(),
                          'Mass': mass.values.tolist(),
                          'dx': dx.values.tolist(),
                          'dy': dy.values.tolist()})

# Performing two-way ANOVA
print('\033[0;30;45m ------ Result for two-way MANOVA ------ \033[0m')
maov = MANOVA.from_formula('dx + dy ~ Needle + Mass', data=dataframe)
print(maov.mv_test())

print('\033[0;30;46m ------ Result for separate two-way ANOVA (dx) ------ \033[0m')
reg = ols('dx ~ Needle + Mass', data=dataframe).fit()
aov = sm.stats.anova_lm(reg, type=2)
print(aov)

print('\033[0;30;46m ------ Result for separate two-way ANOVA (dy) ------ \033[0m')
reg = ols('dy ~ Needle + Mass', data=dataframe).fit()
aov = sm.stats.anova_lm(reg, type=2)
print(aov)

mc1 = pairwise_tukeyhsd(dataframe['dx'], dataframe['Needle'], alpha=0.05)
mc2 = pairwise_tukeyhsd(dataframe['dx'], dataframe['Mass'], alpha=0.05)
mc3 = pairwise_tukeyhsd(dataframe['dy'], dataframe['Needle'], alpha=0.05)
mc4 = pairwise_tukeyhsd(dataframe['dy'], dataframe['Mass'], alpha=0.05)
print('\033[0;30;42m ------ Result for Tukey (dx~Needle) ------ \033[0m')
print(mc1)
print('\033[0;30;42m ------ Result for Tukey (dx~Mass) ------ \033[0m')
print(mc2)
print('\033[0;30;42m ------ Result for Tukey (dy~Needle) ------ \033[0m')
print(mc3)
print('\033[0;30;42m ------ Result for Tukey (dy~Mass) ------ \033[0m')
print(mc4)



