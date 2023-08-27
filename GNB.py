import time

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
)

def TrainAndTestGaussianNB(X_train, X_test, y_train,  y_test):
    
    start = time.time()
    
    NaiveBayes = GaussianNB()
    NaiveBayes.fit(X_train, y_train)
    y_pred_NaiveBayes = NaiveBayes.predict(X_test)
    
    end = time.time() 

    accuracy_NB = accuracy_score(y_pred_NaiveBayes, y_test)
    f1_NB = f1_score(y_pred_NaiveBayes, y_test, average="weighted")
    precision_NB = precision_score(y_pred_NaiveBayes, y_test, average="weighted")
    
    print("\n\no--------------------------------------------------------------------------------o")
    print("Naive bayes precision: ", precision_NB)
    print("Naive bayes accuracy: ", accuracy_NB)
    print("Naive bayes F1 Score: ", f1_NB)
    print("Seconds needed for train and test: ", end - start)