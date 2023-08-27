from sklearn.ensemble import RandomForestClassifier
import time

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    f1_score
    )

import matplotlib.pyplot as plt

def train_and_test_rfc(X_train, X_test, y_train, y_test):
    sample_train_precision_score = 0
    sample_test_precision_score = []
    n_estimators_values = [2, 5, 25, 100, 150, 300, 600]

    
    for n_estimators in n_estimators_values:
        
        start = time.time()

        rfc = RandomForestClassifier(n_estimators=n_estimators)
        rfc.fit(X_train, y_train)

        test_preds = rfc.predict(X_test)
        
        end = time.time()

        train_preds = rfc.predict(X_train)
        
        train_precision = precision_score(y_train, train_preds)
        test_accuracy = accuracy_score(y_test, test_preds)
        test_f1_score = f1_score(y_test, test_preds, average="weighted")
        test_precision = precision_score(y_test, test_preds, average="weighted")
        
        sample_train_precision_score = train_precision
        sample_test_precision_score.append(test_accuracy)

        print('\n\nPrecision of estimators ', n_estimators, ': ', test_precision)
        print('Accuracy of estimators ', n_estimators, ': ', test_accuracy)
        print('F1 Score of estimators ', n_estimators, ': ', test_f1_score)
        print("Seconds needed for train and test: ", end - start)

    # Overfitting test
    plt.bar(['Precisione train', 'Precisione test'], [sample_train_precision_score, sample_test_precision_score[6]])
    plt.ylabel('Precisione')
    plt.title('Verifica overfitting')
    plt.show()

    plt.plot(n_estimators_values, sample_test_precision_score)
    plt.xlabel('Numero alberi')
    plt.title('Confronto sul numero di alberi')
    plt.ylabel('Precisione')
    plt.show()
    
    criterions = ['gini', 'entropy', 'log_loss']
    sample_test_precision_score = []

    print("\no--------------------------------------------------------------------------------o")
    for criterion in criterions:
        
        start = time.time()

        rfc = RandomForestClassifier(n_estimators = 100, criterion=criterion)
        rfc.fit(X_train, y_train)

        test_preds = rfc.predict(X_test)
        
        end = time.time()

        test_accuracy = accuracy_score(y_test, test_preds)
        test_f1_score = f1_score(y_test, test_preds, average="weighted")
        test_precision = precision_score(y_test, test_preds, average="weighted")

        sample_test_precision_score.append(test_precision)

        print('\n\nPrecision of algorithm ', criterion, ' for split quality: ', test_precision)
        print('Accuracy of algorithm ', criterion, ' for split quality: ', test_accuracy)
        print('F1 Score of algorithm ', criterion, ' for split quality: ', test_f1_score)
        print("Seconds needed for train and test: ", end - start)

    plt.bar(criterions, sample_test_precision_score)
    plt.title('Confronto sugli algoritmi per la qualità delle divisioni del dataset')
    plt.ylabel('Precisione')
    plt.ylim(0.7, 1.0)
    plt.show()