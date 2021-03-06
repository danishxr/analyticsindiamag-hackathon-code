
import re
import pandas as pd 
import numpy as np


data=pd.read_csv('predtrainbanghouse.csv')

#data.columns

'''Index(['area_type', 'availability', 'location', 'size', 'society',
       'total_sqft', 'bath', 'balcony', 'price'],
      dtype='object')'''

data.columns=['area_type','availability','location','bhk','society','total_sqft','bath','balcony','price']
#data.balcony.isnull().sum()




def clean (text):
    f=re.split('\W',text)
    c=list(filter(None,f))
    k=sum([float(i) for i in c])/len(c)
    return k

data.total_sqft=data.total_sqft.astype(str)
data.total_sqft=data.total_sqft.apply(lambda x: x.strip())


data.total_sqft=data.total_sqft.apply(lambda x: clean(x))
#data.size=data.size.astype(str)

print(len(data)-data.count())

#data.drop(data.index[4], inplace=True)
data.drop(['society'],axis=1,inplace=True)
data.drop(['location'],axis=1,inplace=True)
#data.drop(['area_type'],axis=1,inplace=True)
#data.drop(['availability'],axis=1,inplace=True)
#data.drop(['size'],axis=1,inplace=True)
#data.drop(['total_sqft'],axis=1,inplace=True)
#data.drop(['bath'],axis=1,inplace=True)
data.drop(['balcony'],axis=1,inplace=True)
data.head()


#finding anomalies 
#data.to_csv('clean.csv')


'''import seaborn as sns
#import matplotlib.pyplot as plt 
sns.boxplot(x=data.total_sqft)

plt.boxplot(data.total_sqft)
#plt.ticklabel_format(useOffset=False)
#plt.ylim(1,1400000000)
plt.show()'''

'''
data = data[data.total_sqft <= 231303]
data = data[data.total_sqft <= 87120]
data = data[data.total_sqft <= 54886]
data = data[data.total_sqft <= 41386]
data = data[data.total_sqft <= 26136]
data = data[data.total_sqft <= 1307000000]
data = data[data.total_sqft <= 10764]
data = data[data.total_sqft <= 10030]
'''
data = data[data.total_sqft <= 7000]
data = data[data.total_sqft >= 480]
data=data[data.bath != 11]
data=data[data.bath != 12]
data=data[data.bath != 13]
data=data[data.bath != 14]
data=data[data.bath != 15]
data=data[data.bath != 18]
data=data[data.bath != 40]


#data.head()


x=data.iloc[:,0:5].values
y=data.iloc[:,5].values

#changing the categorical values 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelx=LabelEncoder()
x[:,0]=labelx.fit_transform(x[:,0])
x[:,1]=labelx.fit_transform(x[:,1])
x[:,2]=labelx.fit_transform(x[:,2])
#x[:,3]=labelx.fit_transform(x[:,3])
#this is to make the categories not a ranking facor by the equation
onehotencoder=OneHotEncoder(categorical_features=[0,1,2,4])
x=onehotencoder.fit_transform(x).toarray()



#model creation
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

from sklearn.linear_model import LinearRegression
modellr=LinearRegression()
modellr.fit(x_train,y_train)
y_pred=modellr.predict(x_test)

trainscore=modellr.score(x_train,y_train)
valscore=modellr.score(x_test,y_test)


from sklearn.metrics import mean_squared_error
from math import sqrt
#from sklearn.metrics import r2_score
RMSE=sqrt(mean_squared_error(y_test, y_pred))
print( "RMSE VALUE>>>",RMSE)
print("Training Accuracy >>>",trainscore*100,"%")
print("Validation Accuracy >>>",valscore*100,"%")
print ('R2 VALUE>>',modellr.score(x_train, y_train), 'ADJUSTED R2 value>>',1 - (1-modellr.score(x_train, y_train))*(len(y_train)-1)/(len(y_train)-x_train.shape[1]-1))

#TESTING ON UNSEEN DATA

testdata=pd.read_csv('predtestbanghouse.csv')
testdata.columns=['area_type','availability','location','bhk','society','total_sqft','bath','balcony','price']
testdata.total_sqft=testdata.total_sqft.astype(str)
testdata.total_sqft=testdata.total_sqft.apply(lambda x: x.strip())


testdata.total_sqft=testdata.total_sqft.apply(lambda x: clean(x))


print(len(testdata)-testdata.count())

#testdata.drop(testdata.index[4], inplace=True)
testdata.drop(['society'],axis=1,inplace=True)
testdata.drop(['location'],axis=1,inplace=True)
#testdata.drop(['area_type'],axis=1,inplace=True)
#testdata.drop(['availability'],axis=1,inplace=True)
#testdata.drop(['size'],axis=1,inplace=True)
#testdata.drop(['total_sqft'],axis=1,inplace=True)
#testdata.drop(['bath'],axis=1,inplace=True)
testdata.drop(['balcony'],axis=1,inplace=True)
testdata.drop(['price'],axis=1,inplace=True)
testdata.head()
#enocding 
#testdata = testdata[testdata.total_sqft <= 7000]
#testdata = testdata[testdata.total_sqft >= 480]

X=testdata.iloc[:,0:5].values


#changing the categorical values 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelX=LabelEncoder()
X[:,0]=labelX.fit_transform(X[:,0])
X[:,1]=labelX.fit_transform(X[:,1])
X[:,2]=labelX.fit_transform(X[:,2])
#X[:,3]=labelX.fit_transform(X[:,3])
#this is to make the categories not a ranking facor by the equation
onehotencoder=OneHotEncoder(categorical_features=[0,1,2,4])
X=onehotencoder.fit_transform(X).toarray()



#export to csv file
testpredictions=modellr.predict(X)
final_solution=pd.DataFrame(testpredictions,columns=['price'])
final_solution.to_excel('LinearRegression.xlsx')
