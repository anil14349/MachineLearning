
#%%
import pandas as pd
import numpy as np
from sklearn import linear_model
import math

df = pd.read_csv('hiring.csv')
df.test_score =df.test_score.fillna(math.floor(df.test_score.median()))

reg = linear_model.LinearRegression()
reg.fit(df[['experience','test_score','interview_score']],df.salary)
reg.predict([[2,9,6]])
# %%
reg.predict([[12,10,10]])


# %%
