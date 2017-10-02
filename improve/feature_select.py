titanic_path = r'..\data\titanic.txt'
import pandas as pd
titanic = pd.read_csv(titanic_path)

y = titanic['survived']
x = titanic.drop(['row.names','name','survived'],axis=1)

x['age'].fillna(x['age'].mean(),inplace=True)
x.fillna("UNKNOWN",inplace=True)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=33)

from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
x_train = vec.fit_transform(x_train.to_dict(orient='record'))
x_test = vec.transform(x_test.to_dict(orient='record'))

print(len(vec.feature_names_))

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
print("all feature:",dt.score(x_test,y_test))

from sklearn import feature_selection
fs = feature_selection.SelectPercentile(feature_selection.chi2,percentile=20)
x_train_fs = fs.fit_transform(x_train,y_train)
dt.fit(x_train_fs,y_train)
x_test_fs = fs.transform(x_test)
print("%20 feature:",dt.score(x_test_fs,y_test))

from sklearn.model_selection import cross_val_score
import numpy as np
precentiles = range(1,100,2)
results = []
for i in precentiles:
    fs = feature_selection.SelectPercentile(feature_selection.chi2,percentile=i)
    x_train_fs = fs.fit_transform(x_train,y_train)
    scores = cross_val_score(dt,x_train_fs,y_train,cv=5)
    results= np.append(results,scores.mean())
print(results)

opt = int(np.where(results == results.max())[0])

print("optimal number of features %d" % precentiles[opt])


