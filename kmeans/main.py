from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
import pandas as pd


bc = load_breast_cancer()
print(bc)
x = scale(bc.data)
print(x)

y = bc.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = KMeans(n_clusters=2, random_state=0)

model.fit(x_train)

prediction = model.predict(x_test)
label = model.labels_
print("Labels:", label)
print("Prediction ", prediction)
print("Accuracy :", accuracy_score(y_test, prediction))
print("Actual :",y_test)