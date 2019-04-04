#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
#importing dataset
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
train.head()
#visulaizng
sex_pivot = train.pivot_table(index ="Sex",values ="Survived")
sex_pivot.plot.bar()
plt.show()
class_pivot= train.pivot_table(index ="Pclass", values= "Survived")
class_pivot.plot.bar()
plt.show()
#elaborating columns
train.Age.describe()
train.Sex.describe()
train["Pclass"].value_counts()
#creating independent and depndent varibales
variables = ["Age","Sex","Pclass","Fare"]
result = ["Survived"]
x = train[variables]
y = train[result]
#checkin for missing data
x["Age"].isnull().sum()
x["Pclass"].isnull().sum()
x["Sex"].isnull().sum()
x["Fare"].isnull().sum()
#removing all the missiing values of age columns with median
x["Age"] = x["Age"].fillna(x["Age"]).median()
x["Sex"] = pd.get_dummies(x["Sex"])
x["Sex"].head()
#importing logistic regression
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x,y)
x.head()
#preprocessing test dataset
x_test = test[variables]
x_test["Sex"] = pd.get_dummies(x_test["Sex"])
x_test["Pclass"].isnull().sum()
x_test["Age"] = x_test["Age"].fillna(x_test["Age"]).median()
x_test["Fare"] = x_test["Fare"].fillna(x_test["Fare"]).median()

y_pred = reg.predict(x_test)
f = open('csvfile.csv','w')

for i in range(len(y_pred)):
    if y_pred[i] <= 0.5:
        y_pred[i] = 0
        f.write('0 \n')
    else:
        y_pred[i] = 1
        f.write('1 \n')
#Give your csv text here.
## Python will convert \n to os.linesep
f.close()
df = pd.read_csv("csvfile.csv")
df.shape
id = [i for i in range(892,1309)]
print(id)
df["PassengerId"] = id
df = df.rename(columns = {"0 ":"Survived"})
print(df)
#checking accuracy of predictionsfrom sklearn.metrics import accuracy_score
accuracy = accuracy_score(y,y_pred)
accuracy
