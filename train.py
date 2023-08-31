import bentoml
from sklearn import svm
from sklearn import datasets

# Load raining dataset

iris = datasets.load_iris()
X,y = iris.data, iris.target

#Train the model
clf = svm.SVC(gamma='scale')
clf.fit(X,y)

#Save the model to the BentoML local model store

saved_model = bentoml.sklearn.save_model("iris_clf",clf)
print(f"Model saved: {saved_model}")


##iris_clf:sq5zj4cicw4pkuc3