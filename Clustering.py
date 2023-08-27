from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from sklearn.metrics import completeness_score, v_measure_score, homogeneity_score

from sklearn.preprocessing import LabelEncoder, StandardScaler

def run_clustering_tests(dataset):
    
    y = dataset['diabetes']

    encoder = LabelEncoder()
    dataset['hypertension_encoded'] = encoder.fit_transform(dataset['hypertension'])
    dataset['heart_disease_encoded'] = encoder.fit_transform(dataset['heart_disease'])
    dataset = dataset.drop(['heart_disease', 'hypertension','gender', 'smoking_history'], axis =1)

    X = dataset.drop('diabetes', axis=1)
    scaler = StandardScaler()
    dataset = scaler.fit_transform(dataset)

    find_best_number_of_clusters(X)
    train_and_test_best_KMeans_model(dataset, y)

def find_best_number_of_clusters(X):
    inertia = []
    k_values = range(1, 25)
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='warn')
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)

    # Traccia il grafico dell'inerzia rispetto al numero di cluster
    plt.plot(k_values, inertia)
    plt.title('Metodo del gomito')
    plt.ylabel('Inerzia')
    plt.xlabel('Cluster')
    plt.show()


def train_and_test_best_KMeans_model(dataset, y):
    
    kmeans = KMeans(n_clusters=2, random_state=42, init='k-means++', max_iter=250)
    kmeans.fit(dataset)
    
    labels = kmeans.labels_

    plt.scatter(dataset[:, 0], dataset[:, 1], c=labels)
    plt.title('Clustering dati')
    plt.show()

    print('\nOmogeneità (k-means++) : ', homogeneity_score(y, labels))
    print('Completezza (k-means++): ', completeness_score(y, labels))
    print('Misura V  (k-means++) : ', v_measure_score(y, labels))

    kmeans = KMeans(n_clusters=2, random_state=42, init='random', max_iter=250)
    kmeans.fit(dataset)
    
    labels = kmeans.labels_

    print('\nOmogeneità (random)  : ', homogeneity_score(y, labels))
    print('Completezza (random): ', completeness_score(y, labels))
    print('Misura V  (random) : ', v_measure_score(y, labels))
