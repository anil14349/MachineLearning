
#%%
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('covidPositiveTel.csv')
X = dataset.iloc[:, 0:1].values
y = dataset.iloc[:, 1].values

# Fitting Polynomial Regression to the dataset

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 6)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)


# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('COVID-19 Telangana Positive Cases')
plt.xlabel('no Of Days')
plt.ylabel('no Of Cases')
plt.show()

# Predicting a new result with Polynomial Regression
print ('Cases by 34th Day : ',lin_reg_2.predict(poly_reg.fit_transform([[34]]))[0])
print ('Cases by 35th Day : ',lin_reg_2.predict(poly_reg.fit_transform([[35]]))[0])
print ('Cases by 36th Day : ',lin_reg_2.predict(poly_reg.fit_transform([[36]]))[0])
print ('Cases by 37th Day : ',lin_reg_2.predict(poly_reg.fit_transform([[37]]))[0])
print ('Cases by 38th Day : ',lin_reg_2.predict(poly_reg.fit_transform([[38]]))[0])
print ('Cases by 39th Day : ',lin_reg_2.predict(poly_reg.fit_transform([[39]]))[0])
print ('Cases by 40th Day : ',lin_reg_2.predict(poly_reg.fit_transform([[40]]))[0])
print ('Cases by 41th Day : ',lin_reg_2.predict(poly_reg.fit_transform([[41]]))[0])

# %%
