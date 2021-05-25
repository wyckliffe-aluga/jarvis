
import lazypredict 
from lazypredict.Supervised import LazyClassifier 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()

X = data.data 
y = data.target 

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=125, test_size=.5)

clf = LazyClassifier(verbose=False, ignore_warnings=True, custom_metric=None)

model, predictions = clf.fit(x_train, x_test, y_train, y_test)

print(model)
print(predictions)