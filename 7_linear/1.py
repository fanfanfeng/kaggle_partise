from sklearn.datasets import load_boston
boston = load_boston()

from sklearn.model_selection import train_test_split
import numpy as np

x = boston.data
y = boston.target

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=33,test_size=0.25)
print('the max target value is ',np.max(boston.target))
print("the min target value is ",np.min(boston.target))
print("the average target value is ",np.mean(boston.target))

from sklearn.preprocessing import StandardScaler
ss_x = StandardScaler()
ss_y = StandardScaler()

x_train = ss_x.fit_transform(x_train)
x_test = ss_x.transform(x_test)
y_train = ss_y.fit_transform(y_train)
y_test = ss_y.transform(y_test)

from sklearn.linear_model import LinearRegression


