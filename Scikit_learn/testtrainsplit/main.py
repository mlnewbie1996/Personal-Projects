from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
iris = datasets.load_iris()

# Split it in features  and labels

X = iris.data
y = iris.target

print(X.shape)
print(y.shape)


# hours of study vs good/bad grades
# train for 8 students and predict for the other 2
# level  of accuracy

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2) # 20 percent of data

print (X_train.shape)
print (X_test.shape)
print (y_train.shape)
print (y_test.shape)

