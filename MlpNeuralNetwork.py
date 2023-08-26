import time

from sklearn.neural_network import MLPClassifier #multilayer perceptron 

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
)

import matplotlib.pyplot as plt

def TestMlpcLayerSize(X_train, y_train, X_test, y_test):

    sample_train_accuracy_score = 0
    sample_test_accuracy_score = 0
    possible_sizes = [(4,4,4),(6,6,6), (8,8,8), (10,10,10)]

    for size in possible_sizes:
    
        start = time.time()

        mlpc = MLPClassifier(hidden_layer_sizes = size, max_iter = 500)
        mlpc.fit(X_train, y_train)
        y_pred = mlpc.predict(X_test)

        end = time.time()

        accuracy_Mlpc = accuracy_score(y_pred, y_test)
        f1_Mlpc = f1_score(y_pred, y_test, average="weighted")
        precision_Mlpc = precision_score(y_pred, y_test, average="weighted")

        print("\no--------------------------------------------------------------------------------o")
        print("Neural network (", str(size), ") precision: ", precision_Mlpc)
        print("Neural network (", str(size), ") accuracy: ", accuracy_Mlpc)
        print("Neural network (", str(size), ") F1 Score: ", f1_Mlpc)
        print("Seconds needed for train and test: ", end - start)


    plt.bar(['Accuratezza train', 'Accuratezza test'], [sample_train_accuracy_score, sample_test_accuracy_score])
    plt.ylabel('Accuracy')
    plt.title('Verifica overfitting')
    plt.show()

def TestMlpcSolver(X_train, y_train, X_test, y_test):

    possible_solvers = ['adam', 'lbfgs', 'sgd']

    for solver in possible_solvers:
    
        start = time.time()

        mlpc = MLPClassifier(hidden_layer_sizes = (8,8,8), solver = solver, max_iter = 500)
        mlpc.fit(X_train, y_train)
        y_pred = mlpc.predict(X_test)

        end = time.time()

        accuracy_Mlpc = accuracy_score(y_pred, y_test)
        f1_Mlpc = f1_score(y_pred, y_test, average="weighted")
        precision_Mlpc = precision_score(y_pred, y_test, average="weighted")

        print("\no--------------------------------------------------------------------------------o")
        print("Neural network (", solver, " solver) precision: ", precision_Mlpc)
        print("Neural network (", solver, " solver) accuracy: ", accuracy_Mlpc)
        print("Neural network (", solver, " solver) F1 Score: ", f1_Mlpc)
        print("Seconds needed for train and test: ", end - start)

def TestMlpcActivation(X_train, y_train, X_test, y_test):

    possible_activations = ['identity', 'logistic', 'relu', 'tanh']

    for activation in possible_activations:
    
        start = time.time()

        mlpc = MLPClassifier(hidden_layer_sizes = (8,8,8), activation = activation, max_iter = 500)
        mlpc.fit(X_train, y_train)
        y_pred = mlpc.predict(X_test)

        end = time.time()

        accuracy_Mlpc = accuracy_score(y_pred, y_test)
        f1_Mlpc = f1_score(y_pred, y_test, average="weighted")
        precision_Mlpc = precision_score(y_pred, y_test, average="weighted")

        print("\no--------------------------------------------------------------------------------o")
        print("Neural network (", activation, " activation) precision: ", precision_Mlpc)
        print("Neural network (", activation, " activation) accuracy: ", accuracy_Mlpc)
        print("Neural network (", activation, " activation) F1 Score: ", f1_Mlpc)
        print("Seconds needed for train and test: ", end - start)

def TrainAndTestBestMLPC(X_train_images, y_train, X_test_images, y_test):
    start = time.time()

    mlpc = MLPClassifier(hidden_layer_sizes=(6,6,6), solver='sgd', activation='identity', max_iter=500)
    mlpc.fit(X_train_images, y_train)
    y_pred = mlpc.predict(X_test_images)

    end = time.time()

    accuracy_Mlpc = accuracy_score(y_pred, y_test)
    f1_Mlpc = f1_score(y_pred, y_test, average="weighted")
    precision_Mlpc = precision_score(y_pred, y_test, average="weighted")

    print("\n\n\nNeural network precision: ", precision_Mlpc)
    print("Neural network accuracy: ", accuracy_Mlpc)
    print("Neural network F1 Score: ", f1_Mlpc)
    print("Seconds needed for train and test: ", end - start)