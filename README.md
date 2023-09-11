# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored
## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.
## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook
## Algorithm:
```
1. Import the required libraries and read the dataframe.
2. Assign hours to X and scores to Y.
3.Implement training set and test set of the dataframe
4.Plot the required graph both for test data and training data.
5.Find the values of MSE , MAE and RMSE.
```
## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Yuvabharathi.B
RegisterNumber:  212222230181
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse) 
```
## Output:
![image](https://github.com/yuvabharathib/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497404/9a1350c0-5109-4be1-b9b4-3aeaec5b16e9) ![image](https://github.com/yuvabharathib/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497404/04a3ca6b-dcf7-4e09-847a-39deba91a11e)
![image](https://github.com/yuvabharathib/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497404/cd5ce0ff-235b-43ea-917c-f385abd7c8e9)
![image](https://github.com/yuvabharathib/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497404/913033c2-7ef7-4546-bced-74e9c86501de)
![image](https://github.com/yuvabharathib/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497404/e72500bf-b019-4e76-a792-136c5ed33fb4)
## Result :
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
