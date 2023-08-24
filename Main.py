import UserInterface
import pandas as pd

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
