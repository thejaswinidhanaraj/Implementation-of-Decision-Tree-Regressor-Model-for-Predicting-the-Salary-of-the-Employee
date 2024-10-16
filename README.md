# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the libraries and read the data frame using pandas. 
2.Calculate the null values present in the dataset and apply label encoder. 
3.Determine test and training data set and apply decison tree regression in dataset.
4.calculate Mean square error,data prediction and r2

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by:THEJASWINI D 
RegisterNumber: 212223110059 
*/
```
```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
data=pd.read_csv("/content/Salary (2).csv")
data.head()
```
![image](https://github.com/user-attachments/assets/b1d101ea-9460-4404-bc8f-e65dbe38e8b2)
```
data.info()
```
![image](https://github.com/user-attachments/assets/34fea81f-a1d8-451c-826c-1227b50da2b8)
```
data.isnull().sum()
```
![image](https://github.com/user-attachments/assets/dc2ad18f-ca61-48ae-bd7f-53955cf2c551)
```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
```
![image](https://github.com/user-attachments/assets/b1c0ecd1-fe14-466b-9571-1a8fb818ce4e)
```
x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=1)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(xtrain,ytrain)
y_pred=dt.predict(xtest)
print(y_pred)
```
![image](https://github.com/user-attachments/assets/73a58380-1283-4ff9-94ec-a78211c418b4)
```
from sklearn import metrics
mse=metrics.mean_squared_error(ytest,y_pred)
print(mse)
```
![image](https://github.com/user-attachments/assets/e041d796-4869-4ccb-8f94-a91a36dbe3ff)
```
r2=metrics.r2_score(ytest,y_pred)
print(r2)
```
![image](https://github.com/user-attachments/assets/1b217086-a008-4331-954e-b10bf0b9e90e)
```
dt.predict([[5,6]])
dt.predict([[5,6]])
plt.figure(figsize=(20, 8))
plot_tree(dt, feature_names=x.columns, filled=True)
plt.show()
```
## Output:
![image](https://github.com/user-attachments/assets/39849a26-3833-429a-b882-7a9342848d57)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
