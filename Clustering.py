from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import completeness_score, homogeneity_score, silhouette_score, v_measure_score

def find_best_number_of_clusters(X_train):
    inertia = []
    k_values = range(1, 25)
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_train)
        inertia.append(kmeans.inertia_)

    # Traccia il grafico dell'inerzia rispetto al numero di cluster
    plt.plot(k_values, inertia, 'bx-')
    plt.title('Metodo del gomito')
    plt.ylabel('Inerzia')
    plt.xlabel('Cluster')
    plt.show()


def train_and_test_best_KMeans_model(X_train, y_train):
    kmeans = KMeans(n_clusters=2, algorithm='full', init='k-means++', max_iter=250,  random_state=42)
    label = kmeans.fit_predict(X_train)
    
    kmeans_labels = np.unique(label)
    X_train = np.array(X_train)

    plt.figure(figsize=(10, 10))
    plt.title('Clustering: ')
    for k in kmeans_labels:
        plt.scatter(X_train[label == k, 0], X_train[label == k, 1], label=k)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='k', label='Cluster mean')
    plt.legend()
    plt.show()
    plt.close()

    print('\nOmogeneità  : ', homogeneity_score(y_train, kmeans.labels_))
    print('Completezza : ', completeness_score(y_train, kmeans.labels_))
    print('V_measure   : ', v_measure_score(y_train, kmeans.labels_))
