import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def print_title():
    print("\no--------------------------------------------o")
    print("| Progetto ICON 22-23: Diabetics classifier  |")
    print("o--------------------------------------------o\n")

def print_pie_chart(values, labels, colors):
    plt.pie(values, labels = labels,colors = colors, autopct='%1.1f%%', startangle=0)
    plt.axis('equal')
    plt.title('Distribuzione dei casi di diabete')
    plt.show() 

def print_dataset_info(dataset):
    print('\nPrime 5 righe del dataset:\n', dataset.head())
    print('\n\nDimensioni del dataset:\n', dataset.shape)
    print('\n\nColonne del dataset:\n', dataset.columns)
    print('\n\nStatistiche dataset:\n', dataset.describe(include='all'))
    print('\n\nAltro:\n', dataset.info())

def compare_using_boxplot(title, x_label, y_label, dataset):
    sns.boxplot(x = x_label, y = y_label, data = dataset)
    plt.title(title)
    plt.show()