import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier

train_file = "train.csv"
test_file = "test.csv"

train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)

x_train = train_data.iloc[:,:-1]
y_train = train_data.iloc[:,-1]

x_test = test_data.iloc[:,:-1]
y_test = test_data.iloc[:,-1]

#print y_train[y_train == "left hand open"].count()
#print y_test[y_test == "left hand open"].count()

noise_variance = 0.1
x_test_noise = np.random.normal(0, noise_variance, x_test.shape)
x_test_noise = 0
x_test += x_test_noise

#print x_train.head()
#print y_train.head()
#print x_train.shape
#print y_train.shape
print x_test.shape
#print y_test.shape
#print x_test_noise.shape

clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(x_train, y_train)
acc = clf.score(x_test, y_test)
print "Test accuracy: ", acc

pred = clf.predict(x_test)
# Get [0] since empty second element is returned in tuple
misclassified = np.where(pred != y_test)[0] 
print misclassified
print misclassified.shape
print y_test[misclassified]

