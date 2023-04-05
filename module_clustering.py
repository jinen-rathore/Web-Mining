import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def optimalKvalue(df):

    X = df[["latitude","longitude"]]
    max_k = 10## iterations
    distortions = [] 
    for i in range(1, max_k+1):
        if len(X) >= i:
            model = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
            model.fit(X)
        distortions.append(model.inertia_)## best k: the lowest derivative
    k = [i*100 for i in np.diff(distortions,2)].index(min([i*100 for i 
        in np.diff(distortions,2)]))## plot
    fig, ax = plt.subplots()
    ax.plot(range(1, len(distortions)+1), distortions)
    ax.axvline(k, ls='--', color="red", label="k = "+str(k))
    ax.set(title='The Elbow Method', xlabel='Number of clusters', 
        ylabel="Distortion")
    ax.legend()
    ax.grid(True)
    plt.show()

def kMeans_clustering(df, k):
    map = df[["longitude", "latitude"]]
    kmeans = KMeans(n_clusters=k, n_init='auto')
    kmeans.fit(map)

    clusters = kmeans.fit_predict(map)

    cluster_data = df.copy()
    cluster_data["Kclusters"] = clusters

    plt.figure(figsize=(8,8))
    plt.scatter(cluster_data["longitude"],cluster_data["latitude"],c=cluster_data['Kclusters'],cmap='rainbow')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f"Data with {k} clusters") 
    plt.xlim(-180,180)
    plt.ylim(-90, 90)
    plt.show()

def clustering_program(df):
    option = int(input("""What do you want to do:
    1\t Check the optimal k value:
    2\t k-means clustering: 
    enter answer(1/2): """))

    if option == 1:
        optimalKvalue(df)
    elif option == 2:
        kvalue = int(input("Please enter k value: "))
        kMeans_clustering(df, kvalue)
    else:
        print('\nSorry, this is not an option, we will return to the main program') 
        
        

    




