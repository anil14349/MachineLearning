

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values


#encoding ccategorical data
#enccoding the Independant variables Xn
from sklearn.preprocessing import OneHotEncoder 
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('State', OneHotEncoder(),[3])],remainder = 'passthrough')
X = np.array(ct.fit_transform(X), dtype = np.float)


#Remove dummy variable trap
X = X[:,1:] # remove the index columnclear


# split the data set into training & test set
from  sklearn.model_selection  import  train_test_split
X_train , X_test ,y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)



# Fitting multiple linear regression to the training set
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,y_train)    


# predict the test set Result
y_pred = reg.predict(X_test)

"""
# backword elimination model
import statsmodels.api as sm
X = np.append(arr=np.ones((50,1)).astype(int),values=X ,axis=1)
# iteration 1
X_opt = X[:,[0,1,2,3,4,5]]
reg_OLS = sm.OLS(y , X_opt).fit()
#print(reg_OLS.summary())
# iteration 2
X_opt = X[:,[0,1,3,4,5]]
reg_OLS = sm.OLS(y , X_opt).fit()
#print(reg_OLS.summary()) 
# iteration 3
X_opt = X[:,[0,3,4,5]]
reg_OLS = sm.OLS(y , X_opt).fit()
#print(reg_OLS.summary()) 
# iteration 4
X_opt = X[:,[0,3,5]]
reg_OLS = sm.OLS(y , X_opt).fit()
#print(reg_OLS.summary()) 
# iteration 5
X_opt = X[:,[0,3]]
reg_OLS = sm.OLS(y , X_opt).fit()
print(' Manual Iteration : ',reg_OLS.summary()) 
"""


# Backward Elimination with p-values only:


def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        print("PValue : ",maxVar)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    print(regressor_OLS.summary())
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4]]
X_Modeled = backwardElimination(X_opt, SL)
#print('Backward Elimination with p-values only ',X_Modeled)

"""
# Backward Elimination with p-values and Adjusted R Squared :
def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = np.zeros((50,6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4]]
X_Modeled = backwardElimination(X_opt, SL)
print(' Backward Elimination with p-values and Adjusted R Squared : ',X_Modeled)
"""