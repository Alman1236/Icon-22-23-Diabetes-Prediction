import time

from sklearn.neural_network import MLPClassifier #multilayer perceptron 

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
)

import matplotlib.pyplot as plt

def TestMlpcLayerSize(X_train, X_test, y_train,  y_test):

    sample_train_precision_score = []
    sample_test_precision_score = []
    possible_sizes = [(4,4,4),(6,6,6), (8,8,8), (10,10,10)]

    print("\no--------------------------------------------------------------------------------o")
    for size in possible_sizes:
    
        start = time.time()

        mlpc = MLPClassifier(hidden_layer_sizes = size, max_iter = 500)
        mlpc.fit(X_train, y_train)
        y_pred = mlpc.predict(X_test)

        end = time.time()

        y_train_pred = mlpc.predict(X_train)
        precision_train_Mlpc = precision_score(y_train_pred, y_train, average="weighted")

        accuracy_Mlpc = accuracy_score(y_pred, y_test)
        f1_Mlpc = f1_score(y_pred, y_test, average="weighted")
        precision_Mlpc = precision_score(y_pred, y_test, average="weighted")

        sample_test_precision_score.append(precision_Mlpc)
        sample_train_precision_score.append(precision_train_Mlpc)

        print("\n\nNeural network (", str(size), ") precision: ", precision_Mlpc)
        print("Neural network (", str(size), ") accuracy: ", accuracy_Mlpc)
        print("Neural network (", str(size), ") F1 Score: ", f1_Mlpc)
        print("Seconds needed for train and test: ", end - start)


    plt.bar(['Precisione train', 'Precisione test'], [sample_train_precision_score[0], sample_test_precision_score[0]])
    plt.ylabel('Precisione')
    plt.title('Verifica overfitting')
    plt.show()

    sizes = ['4x4x4','6x6x6','8x8x8','10x10x10' ]
    plt.bar(sizes, sample_test_precision_score)

    plt.title('Confronto sulla dimensione della rete')
    plt.ylabel('Precisione')
    plt.ylim(0.7, 1.0)
    plt.show()

def TestMlpcSolver(X_train, X_test, y_train,  y_test):

    possible_solvers = ['adam', 'lbfgs', 'sgd']
    sample_test_precision_score = []

    print("\no--------------------------------------------------------------------------------o")
    for solver in possible_solvers:
    
        start = time.time()

        mlpc = MLPClassifier(hidden_layer_sizes = (8,8,8), solver = solver, max_iter = 500)
        mlpc.fit(X_train, y_train)
        y_pred = mlpc.predict(X_test)

        end = time.time()

        accuracy_Mlpc = accuracy_score(y_pred, y_test)
        f1_Mlpc = f1_score(y_pred, y_test, average="weighted")
        precision_Mlpc = precision_score(y_pred, y_test, average="weighted")

        sample_test_precision_score.append(precision_Mlpc)

        print("\n\nNeural network (", solver, " solver) precision: ", precision_Mlpc)
        print("Neural network (", solver, " solver) accuracy: ", accuracy_Mlpc)
        print("Neural network (", solver, " solver) F1 Score: ", f1_Mlpc)
        print("Seconds needed for train and test: ", end - start)

    plt.bar(possible_solvers, sample_test_precision_score)
    plt.ylim(0.7, 1.0)
    plt.title('Confronto sull\' algoritmo per la computazione dei pesi')
    plt.ylabel('Precisione')
    plt.show()

def TestMlpcActivation(X_train, X_test, y_train,  y_test):

    possible_activations = ['identity', 'logistic', 'relu', 'tanh']
    sample_test_precision_score = []

    print("\no--------------------------------------------------------------------------------o")
    for activation in possible_activations:
    
        start = time.time()

        mlpc = MLPClassifier(hidden_layer_sizes = (8,8,8), activation = activation, max_iter = 500)
        mlpc.fit(X_train, y_train)
        y_pred = mlpc.predict(X_test)

        end = time.time()

        accuracy_Mlpc = accuracy_score(y_pred, y_test)
        f1_Mlpc = f1_score(y_pred, y_test, average="weighted")
        precision_Mlpc = precision_score(y_pred, y_test, average="weighted")

        sample_test_precision_score.append(precision_Mlpc)

        print("\n\nNeural network (", activation, " activation) precision: ", precision_Mlpc)
        print("Neural network (", activation, " activation) accuracy: ", accuracy_Mlpc)
        print("Neural network (", activation, " activation) F1 Score: ", f1_Mlpc)
        print("Seconds needed for train and test: ", end - start)

    plt.bar(possible_activations, sample_test_precision_score)
    plt.title('Confronto sulla funzione di attivazione per gli strati nascosti')
    plt.ylabel('Precisione')
    plt.ylim(0.7, 1.0)
    plt.show()

def TrainAndTestBestMLPC(X_train, X_test, y_train,  y_test):
    start = time.time()

    mlpc = MLPClassifier(hidden_layer_sizes=(6,6,6), solver='lbfgs', activation='relu', max_iter=500)
    mlpc.fit(X_train, y_train)
    y_pred = mlpc.predict(X_test)

    end = time.time()

    accuracy_Mlpc = accuracy_score(y_pred, y_test)
    f1_Mlpc = f1_score(y_pred, y_test, average="weighted")
    precision_Mlpc = precision_score(y_pred, y_test, average="weighted")

    print("\n\nNeural network precision: ", precision_Mlpc)
    print("Neural network accuracy: ", accuracy_Mlpc)
    print("Neural network F1 Score: ", f1_Mlpc)
    print("Seconds needed for train and test: ", end - start)