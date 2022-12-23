from sklearn.cluster import KMeans, AgglomerativeClustering

from data import load_data
from plots import show_plot


if __name__ == '__main__':
    data = load_data('../Economic Data - 9 Countries (1980-2020).csv')

    labels = {
        'x': 'Rate',
        'y': 'Price',
    }

    linkage = AgglomerativeClustering(
        n_clusters=4,
        metric='euclidean',
        linkage='ward',
    )
    linkage.fit(data)

    kmeans = KMeans(
        n_clusters=4,
        n_init='auto',
        max_iter=1000,
        algorithm='elkan'
    )
    kmeans.fit(data)

    show_plot(linkage, labels, data)
    show_plot(kmeans, labels, data)
