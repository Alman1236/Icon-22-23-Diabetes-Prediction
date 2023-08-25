import UserInterface
import pandas as pd
import RandomForest
import Clustering

from sklearn.model_selection import train_test_split

UserInterface.print_title()

dataset = pd.read_csv('dataset.csv')

UserInterface.print_dataset_info(dataset)

diabetic_count = dataset['diabetes'].value_counts()

labels = ['Non diabetici', 'Diabetici']
values = diabetic_count.values
colors = ['#16ff00', '#ff0000']
UserInterface.print_pie_chart(values, labels, colors)

#Si potrebbe decidere di mostrare quanti degli affetti da malattie cardiache hanno il diabete, in percentuale, con un grafico a torta
#Lo stesso si potrebbe fare con l'ipertensione

data_for_boxplot = dataset.drop(['gender', 'age', 'smoking_history'], axis=1)
UserInterface.compare_using_boxplot(title = 'Distribuzione BMI in relazione al diabete', x_label= 'diabetes', y_label='bmi', dataset = data_for_boxplot)
UserInterface.compare_using_boxplot(title = 'Distribuzione emoglobina glicata in relazione al diabete', x_label= 'diabetes', y_label='HbA1c_level', dataset = data_for_boxplot)
UserInterface.compare_using_boxplot(title = 'Distribuzione glucosio nel sangue in relazione al diabete', x_label= 'diabetes', y_label='blood_glucose_level', dataset = data_for_boxplot)

dataset['smoker_bool'] = 0
dataset['male_bool'] = 0
for i in range(len(dataset['smoker_bool'])):

    smoker = str(dataset['smoking_history'][i]).lower()
    gender = str(dataset['gender'][i]).lower()

    if(smoker == 'former' or smoker == 'current' or smoker == 'not current'):
        dataset.at[i, 'smoker_bool'] = 1

    if(gender == 'male'):
        dataset.at[i, 'male_bool'] = 1


X_train, X_test, y_train, y_test = train_test_split(
    dataset.drop(['diabetes', 'smoking_history','gender'], axis=1),
    dataset['diabetes'], 
    test_size=0.2, 
    stratify=dataset['diabetes'], 
    random_state=42)

#RandomForest.train_and_test_rfc(X_train, X_test, y_train, y_test)

Clustering.train_and_test_best_KMeans_model(X_train, y_train)