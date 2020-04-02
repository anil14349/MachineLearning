#%% 
# imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from  sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


#%% 

# read the csv
df = pd.read_csv('canada_per_capita_income.csv')
X = df.iloc[:,:-1].values
y = df.iloc[:,1].values

X
#%%
# split the dataset into test and train set
X_train, X_test ,y_train, y_test = train_test_split(X , y, test_size= 1/3 , random_state=0)

# LenearRegression
reg = LinearRegression()
reg.fit(X_train,y_train)

# predicting the Test set results
y_pred = reg.predict(X_test)

"""
# visualising the Training set results

"""
plt.scatter(X_train,y_train,color= 'red',marker='+')
plt.plot(X_train,reg.predict(X_train), color='blue')
plt.title('Year Vs income (Training Set)')
plt.xlabel('Year')
plt.ylabel('Income')
plt.show()
# %%
"""
# visualising the Test set results
"""

plt.scatter(X_test,y_test,color= 'red')
plt.plot(X_train,reg.predict(X_train), color='blue')
plt.title('Salary Vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# %%


# %%
