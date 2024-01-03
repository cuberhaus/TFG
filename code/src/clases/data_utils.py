import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
import pandas as pd


def get_all_bounding_boxes(dataset):
    all_bounding_boxes = []
    for _, target in dataset:
        boxes = target['boxes'].cpu().numpy()
        for box in boxes:
            all_bounding_boxes.append(box)
    return all_bounding_boxes


def calculate_wcss(data, max_k=10):
    wcss = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    return wcss


def plot_elbow(wcss):
    plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(wcss) + 1), wcss, marker='o')
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('WCSS')
    plt.xticks(range(1, len(wcss) + 1))
    plt.grid(True)
    plt.show()


def calculate_silhouette_scores(data, max_k=10):
    silhouette_scores = []
    for k in range(2, max_k + 1):  # Silhouette score only makes sense for 2 or more clusters
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
        cluster_labels = kmeans.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    return silhouette_scores


def plot_silhouette_scores(silhouette_scores):
    plt.figure(figsize=(10, 8))
    plt.plot(range(2, len(silhouette_scores) + 2), silhouette_scores, marker='o')
    plt.title('Silhouette Scores for Different Numbers of Clusters')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.xticks(range(2, len(silhouette_scores) + 2))
    plt.grid(True)
    plt.show()


def get_bbox_centers(bounding_boxes):
    centers = []
    for bbox in bounding_boxes:
        x_min, y_min, x_max, y_max = bbox
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        centers.append([x_center, y_center])
    return centers


def cluster_bounding_boxes(bbox_centers, n_clusters=3):
    bbox_centers = np.array(bbox_centers)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(bbox_centers)
    centers = kmeans.cluster_centers_
    return centers, kmeans.labels_


def plot_cluster_centers(centers):
    plt.figure(figsize=(10, 8))
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', label='Cluster Centers')

    # Annotate the cluster centers
    for i, center in enumerate(centers):
        plt.annotate(f'Center {i + 1}', (center[0], center[1]), textcoords="offset points", xytext=(0, 10), ha='center')

    plt.title('Cluster Centers')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.legend()
    plt.show()


def plot_cluster_centers_with_bbox_centers(centers, bbox_centers):
    if isinstance(bbox_centers, list):
        bbox_centers = np.array(bbox_centers)
    if isinstance(centers, list):
        centers = np.array(centers)

    plt.scatter(bbox_centers[:, 0], bbox_centers[:, 1], c='blue', marker='o', label='Bounding Box Centers')

    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', label='Cluster Centers')

    # Annotate the cluster centers
    for i, center in enumerate(centers):
        plt.annotate(f'Cluster Center {i + 1}', (center[0], center[1]), textcoords="offset points", xytext=(0, 10),
                     ha='center')

    plt.title('Bounding Box Centers and Cluster Centers')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.legend()
    plt.show()


def plot_density_of_bbox_centers(bbox_centers):
    if isinstance(bbox_centers, list):
        bbox_centers = np.array(bbox_centers)

    # Needed for Seaborn
    df_centers = pd.DataFrame(bbox_centers, columns=['x_center', 'y_center'])

    plt.figure(figsize=(10, 8))
    ax = sns.kdeplot(data=df_centers, x='x_center', y='y_center', fill=True, cmap="Blues")

    # Create a color bar for the scale
    norm = plt.Normalize(df_centers.values.min(), df_centers.values.max())
    sm = plt.cm.ScalarMappable(cmap="Blues", norm=norm)
    sm.set_array([])

    # Add a color bar as the legend
    color_bar = plt.colorbar(sm, ax=ax)
    color_bar.set_label('Density')

    plt.title('Density Plot of Bounding Box Centers')
    plt.xlabel('X Center')
    plt.ylabel('Y Center')


def plot_density_of_bbox_centers_with_clusters(bbox_centers, cluster_centers):
    if isinstance(bbox_centers, list):
        bbox_centers = np.array(bbox_centers)

    # Needed for Seaborn
    df_centers = pd.DataFrame(bbox_centers, columns=['x_center', 'y_center'])

    plt.figure(figsize=(10, 8))
    ax = sns.kdeplot(data=df_centers, x='x_center', y='y_center', fill=True, cmap="Blues", legend=True)

    norm = plt.Normalize(df_centers.values.min(), df_centers.values.max())
    sm = plt.cm.ScalarMappable(cmap="Blues", norm=norm)
    sm.set_array([])

    color_bar = plt.colorbar(sm,ax=ax)
    color_bar.set_label('Density')

    if isinstance(cluster_centers, list):
        cluster_centers = np.array(cluster_centers)
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x', label='Cluster Centers')

    plt.title('Density Plot of Bounding Box Centers with Cluster Centers')
    plt.xlabel('X Center')
    plt.ylabel('Y Center')
    plt.legend()

    plt.show()
