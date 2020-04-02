import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from  sklearn.model_selection import train_test_split

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

#split the dataset clear
X_train, X_test ,y_train, y_test = train_test_split(X , y, test_size= 1/3 , random_state=0)


# fitting the simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)


# predicting the Test set results
y_pred = regressor.predict(X_test)


# visualising the Training set results
plt.scatter(X_train,y_train,color= 'red')
plt.plot(X_train,regressor.predict(X_train), color='blue')
plt.title('Salary Vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# visualising the Test set results
plt.scatter(X_test,y_test,color= 'red')
plt.plot(X_train,regressor.predict(X_train), color='blue')
plt.title('Salary Vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
