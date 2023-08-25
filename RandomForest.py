from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    f1_score
    )

import matplotlib.pyplot as plt

def train_and_test_rfc(X_train, X_test, y_train, y_test):
    train_scores = []
    test_scores = []
    n_estimators_values = [25, 50, 100, 150, 200, 300]

    for n_estimators in n_estimators_values:
        rfc = RandomForestClassifier(n_estimators=n_estimators)
        rfc.fit(X_train, y_train)

        train_preds = rfc.predict(X_train)
        test_preds = rfc.predict(X_test)

        train_accuracy = accuracy_score(y_train, train_preds)
        
        test_accuracy = accuracy_score(y_test, test_preds)
        test_f1_score = f1_score(y_test, test_preds, average="weighted")
        test_precision = precision_score(y_test, test_preds, average="weighted")
        
        train_scores.append(train_accuracy)
        test_scores.append(test_accuracy)

        print('\n\nPrecision of estimators ', n_estimators, ': ', test_precision)
        print('Accuracy of estimators ', n_estimators, ': ', test_accuracy)
        print('F1 Score of estimators ', n_estimators, ': ', test_f1_score)

    # Overfitting test
    plt.bar(['Accuratezza train', 'Accuratezza test'], [train_scores[0], test_scores[0]])
    plt.ylabel('Accuracy')
    plt.title('Verifica overfitting')
    plt.show()

    criterions = ['gini', 'entropy', 'log_loss']
    for criterion in criterions:
        rfc = RandomForestClassifier(n_estimators = 25, criterion=criterion)
        rfc.fit(X_train)

        test_preds = rfc.predict(X_test)
        
        test_accuracy = accuracy_score(y_test, test_preds)
        test_f1_score = f1_score(y_test, test_preds, average="weighted")
        test_precision = precision_score(y_test, test_preds, average="weighted")
        
        test_scores.append(test_accuracy)

        print('\n\nPrecision of algorithm ', criterion, ' for split quality: ', test_precision)
        print('Accuracy of algorithm ', criterion, ' for split quality: ', test_accuracy)
        print('F1 Score of algorithm ', criterion, ' for split quality: ', test_f1_score)