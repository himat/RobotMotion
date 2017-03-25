import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier

train_file = "basic_left_train.csv"
test_file = "basic_left_test.csv"

train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)

print type(train_data)

x_train = train_data.iloc[:,:-1]
y_train = train_data.iloc[:,-1]

x_test = test_data.iloc[:,:-1]
y_test = test_data.iloc[:,[-1]]

print x_train.head()
print y_train.head()
print x_train.shape
print y_train.shape
print x_test.shape
print y_test.shape

clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(x_train, y_train)
acc = clf.score(x_test, y_test)
print "Test accuracy: ", acc

print x_test.iloc[0]
print y_test.iloc[0]

one_input = x_test.iloc[[0],:]
print one_input.shape
predicted = clf.predict(one_input)
print "predicted: ", predicted
