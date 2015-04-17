__author__ = 'kevintandean'
import numpy as np


import pandas as pd

path = 'all data full descriptors.txt'

df = pd.read_csv(path,sep='\t')

descriptors = df.iloc[:,15:1890]
logBB = df['log BB']
logBB = logBB.apply(lambda x: -1 if x < 0 else x)

#join table
table = descriptors.join(logBB)

#split training


print df.head()
print
print df.PubchemFP847
values = df.iloc[[2]].values
print values