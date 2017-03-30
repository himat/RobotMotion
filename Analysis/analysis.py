import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier

np.random.seed(12)

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

#print x_train.head()
#print y_train.head()
print "Train data shape: ", x_train.shape
#print y_train.shape
print "Test data shape: ", x_test.shape
#print y_test.shape
#print x_test_noise.shape

noise_variances = np.linspace(0.0, 0.5, 6)
print noise_variances

print "Training"
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(x_train, y_train)

print "Testing"
acc_results = []
for noise_var in noise_variances:
    print "Testing with noise var=", noise_var
    
    if noise_var == 0:
        noise = 0
    else: # Can't use np normal with 0 var
        noise = np.random.normal(0, noise_var, x_test.shape)
    x_test_noised = x_test + noise

    acc = clf.score(x_test_noised, y_test)
    print "Test accuracy: ", acc

    acc_results.append((noise_var, acc))

    pred = clf.predict(x_test_noised)
    # Get [0] since empty second element is returned in tuple
    misclassified = np.where(pred != y_test)[0] 
    print misclassified
    print y_test[misclassified].head()
print acc_results
output_format = "%-10s: %-10s"
print output_format % ("Noise", "Test Accuracy")
output_format = "%-10f: %-10f"
for (noise, acc) in acc_results:
    print output_format % (noise, acc)

print "Single sample"
sample = 141
var = 0.2
x_test_mod = x_test + np.random.normal(0, var, x_test.shape)
ex1 = x_test_mod.iloc[sample,:]
#print ex1
print "Exp: ", y_test[sample]
predicted = clf.predict(ex1.values.reshape(1,-1))
print "Produced: ", predicted
print "Same?: ", y_test[sample] == predicted
probs = clf.predict_proba(ex1.values.reshape(1,-1))
print "Classes: ", clf.classes_
print "Probs: ", probs


