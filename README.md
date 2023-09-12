# EXP 2: Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored
## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

##Equipments Required:
Hardware – PCs
Anaconda – Python 3.7 Installation / Jupyter notebook
## Algorithm
### STEP 1:
Import the needed packages

### STEP 2:
Assigning hours To X and Scores to Y

### STEP 3:
Plot the scatter plot

### STEP 4 :
Use mse,rmse,mae formmula to find

### Program:
Developed by: Yuvabharathi.B

RegisterNumber:212222230181

### Program to implement the simple linear regression model for predicting the marks scored.

### df.head()
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/student_scores.csv')
df.head()
```
### df.tail()
```
df.tail()
```
### Array value of X
```
X = df.iloc[:,:-1].values
X
```
### Array value of Y
```
Y = df.iloc[:,1].values
Y
```
### Values of Y prediction
```
Y_pred
```
### Array values of Y test
```
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
```
### Training Set Graph
```
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="purple")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
### Test Set Graph
```
plt.scatter(X_test,Y_test,color="grey")
plt.plot(X_test,regressor.predict(X_test),color="blue")
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
### Values of MSE,MAE AND RMSE
```
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```
## Output:
### df.head()
![image](https://github.com/yuvabharathib/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497404/8cca104c-ec18-4159-a41d-b85d1f4c75d0)


### df.tail()
![image](https://github.com/yuvabharathib/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497404/e2a3d5b3-d910-480a-bdba-b1bc3feeda15)


### Array value of X
![image](https://github.com/yuvabharathib/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497404/26ec42bc-2658-4a60-975c-dd2b47e6f670)


### Array value of Y
![image](https://github.com/yuvabharathib/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497404/4067c173-c46d-4e9d-ba4b-ab6ded42c060)


### Values of Y prediction
![image](https://github.com/yuvabharathib/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497404/c106d590-d799-45f3-9937-8b739a4e1cfc)


### Array values of Y test
![image](https://github.com/yuvabharathib/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497404/37e6652e-2a09-4b4b-8ebb-82c39c3ff5f1)


### Training Set Graph
![image](https://github.com/yuvabharathib/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497404/a8574185-8810-4f07-bb1e-3644049dd098)


### Test Set Graph
![image](https://github.com/yuvabharathib/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497404/8cbea21c-058a-4ccd-aae7-6ff72ebc57e8)


### Values of MSE,MAE AND RMSE
![image](https://github.com/yuvabharathib/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497404/2f82d59e-ce09-4f66-9aa9-d860625b11e0)![image](https://github.com/yuvabharathib/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497404/2331e7a4-a013-461c-a13b-2cd2c5e00aa1)![image](https://github.com/yuvabharathib/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497404/fc880418-4dbc-4225-ba35-d3a0c4bdf66b)
## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
