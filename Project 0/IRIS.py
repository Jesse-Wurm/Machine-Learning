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

