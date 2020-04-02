
#%%
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn import linear_model


#%%
df = pd.read_csv('canada_per_capita_income.csv')
df


#%%
%matplotlib inline
plt.xlabel('Year')
plt.ylabel('income in US $')
plt.scatter(df.year,df.income,color = 'red',marker='+')


#%%
# create regression model
reg = linear_model.LinearRegression()
reg.fit(df[['year']],df.income)
#predict the value

reg.predict([[2020]])

#%%
reg.coef_
#%%
reg.intercept_

# %%
# y = mx+b

reg.coef_ * 2020 +   reg.intercept_   


# %%
# generate plot
d = pd.read_csv('canada_per_capita_income-predict.csv')
d.head(3)

# %%
p = reg.predict(d)

# %%
