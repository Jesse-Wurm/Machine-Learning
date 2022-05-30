#import libraries
# I had to change how the dataset was imported as I was using an other example
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

#Fixed the import libraries to work with sklearn

# load the IRIS dataset
iris = datasets.load_iris()

# import features and labels

#print tehe IRIS feature names
print(iris.feature_names)

# that will print the names of the IRIS features, next we are going print the target names

#print the IRIS target names
print(iris.target_names)

#lets look at that dataset

#print the whole dataset
print(iris.data)

#thats a lot of data, now lets look at the target "Class labels"
print(iris.target)

#now that we've seen the data, lets assign it to variables
x = iris.data
y = iris.target

#Lets print the data dimensions
print(x.shape)
print(y.shape)

#Building a Classification Mdel with Random Forest
clf = RandomForestClassifier()

print(clf.fit(x, y))


print(clf.feature_importances_)

#predictions

x[0]

print(clf.predict([[5.1, 3.5, 1.4, 0.2]]))
print(clf.predict(x[[0]]))
print(clf.predict_proba(x[[0]]))
clf.fit(iris.data, iris.target_names[iris.target])

#split the data into 80/20 train/test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
x_train.shape, y_train.shape, x_test.shape, y_test.shape

#rebuild the model
clf.fit(x_train, y_train)
print(clf.predict([[5.1, 3.5, 1.4, 0.2]]))
print(clf.predict_proba([[5.1, 3.5, 1.4, 0.2]]))
print(clf.predict(x_test))
print(y_test)

#model preformance evaluation
print(clf.score(x_test, y_test))