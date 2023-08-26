from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

from sklearn.metrics import precision_score

import time

def train_and_test_knn(X_train, X_test, y_train, y_test):
    
    test_scores = []
    k_values = range(1, 26) 

    for k in k_values:
        
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)

        y_pred_knn = knn.predict(X_test)
        test_score = precision_score(y_pred_knn, y_test, average="weighted")

        test_scores.append(test_score)

    plt.plot(k_values, test_scores)
    plt.xlabel('Numero vicini')
    plt.title('Valutazione in base al numero di vicini')
    plt.ylabel('Precisione')
    plt.show()

    test_scores = []
    time_used = []
    algorithms = ['auto', 'ball_tree', 'kd_tree', 'brute'] 

    for algorithm in algorithms:
        
        start = time.time()
        knn = KNeighborsClassifier(n_neighbors=10, algorithm=algorithm)
        knn.fit(X_train, y_train)

        y_pred_knn = knn.predict(X_test)

        end = time.time()

        test_score = precision_score(y_pred_knn, y_test, average="weighted")

        time_used.append(end - start)
        test_scores.append(test_score)

    plt.bar(algorithms, test_scores)
    plt.title('Confronto algoritmi usati per computare il vicino meno distante')
    plt.ylabel('Precision')
    plt.show()

    plt.bar(algorithms, time_used)
    plt.title('Confronto sul tempo degli algoritmi')
    plt.ylabel('Tempo impiegato')
    plt.show()
