import os
import pickle
import pandas as pd

from pulsar import Function
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def train_iris_model():
    # If we already have an existing model file, we can load it right away
    if os.path.exists("model.pkl"):
        print("We are loading a pre-existing model")
        return pickle.load(open("model.pkl", 'rb'))

    # Read in the iris data, split into a 20% test component
    iris = pd.read_csv("https://datahub.io/machine-learning/iris/r/iris.csv")
    train, test = train_test_split(iris, test_size=0.2, stratify=iris['class'])

    # Get the training data
    X_train = train[['sepalwidth', 'sepallength', 'petalwidth', 'petallength']]
    y_train = train['class']

    # Get the test data
    X_test = test[['sepalwidth', 'sepallength', 'petalwidth', 'petallength']]
    y_test = test['class']

    # Train the model 
    model = DecisionTreeClassifier(max_depth=3, random_state=1)
    model.fit(X_train, y_train)

    # Dump the model object to a file
    pickle.dump(model, open("model.pkl", 'wb'))

    return model


class IrisPredictionFunction(Function):
    # No initialization code needed
    def __init__(self):
        pass

    def process(self, input, context):
        # Convert the input ratio to a float, if it isn't already
        flower_parameters = [float(x) for x in input.split(",")]

        # Get the prediction
        model = train_iris_model()
        return model.predict([flower_parameters])[0]


class IrisPredictionFunctionBulk(Function):
    # No initialization code needed
    def __init__(self):
        pass

    def process(self, input, context):
        # Convert the input parameters to floats, if they aren't already
        flower_parameters_str = input.split(":")
        flower_parameters_split = [x.split(",") for x in flower_parameters_str]
        flower_parameters_float = [[float(x) for x in y] for y in flower_parameters_split]

        # Get the prediction
        model = train_iris_model()
        return ", ".join(model.predict(flower_parameters_float))
