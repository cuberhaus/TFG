import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def get_all_bounding_boxes(dataset):
    """
    Get all bounding boxes from a dataset.
    """
    all_bounding_boxes = []
    for _, target in dataset:
        boxes = target['boxes'].cpu().numpy()
        for box in boxes:
            all_bounding_boxes.append(box)
    return all_bounding_boxes


def calculate_wcss(data, max_k=10):
    """
    Calculate the within-cluster sum of squares (WCSS) for different numbers of clusters.
    """
    wcss = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    return wcss


def plot_elbow(wcss):
    """
    Plot the within-cluster sum of squares (WCSS) for different numbers of clusters.
    """
    plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(wcss) + 1), wcss, marker='o')
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('WCSS')
    plt.xticks(range(1, len(wcss) + 1))
    plt.grid(True)
    plt.show()


def calculate_silhouette_scores(data, max_k=10):
    """
    Calculate silhouette scores for different numbers of clusters.

    Parameters:
    - data: The data used for clustering.
    - max_k: Maximum number of clusters to try.

    Returns:
    - A list of silhouette scores corresponding to the number of clusters.
    """
    silhouette_scores = []
    for k in range(2, max_k + 1):  # Silhouette score is only defined for 2 or more clusters
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
        cluster_labels = kmeans.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    return silhouette_scores


def plot_silhouette_scores(silhouette_scores):
    """
    Plot the silhouette scores for different numbers of clusters.

    Parameters:
    - silhouette_scores: List of silhouette scores.
    """
    plt.figure(figsize=(10, 8))
    plt.plot(range(2, len(silhouette_scores) + 2), silhouette_scores, marker='o')
    plt.title('Silhouette Scores for Different Numbers of Clusters')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.xticks(range(2, len(silhouette_scores) + 2))
    plt.grid(True)
    plt.show()


def cluster_bounding_boxes(bounding_boxes, n_clusters=3):
    """
    Cluster bounding boxes using K-means clustering.
    """
    # Convert from [x1, y1, x2, y2] to [x, y, width, height]
    data = []
    for bbox in bounding_boxes:
        x, y, x2, y2 = bbox
        width = x2 - x
        height = y2 - y
        data.append([x, y, width, height])
    data = np.array(data)

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)

    # Get cluster centers
    centers = kmeans.cluster_centers_

    return centers, kmeans.labels_


def plot_cluster_centers(centers):
    """
    Plot the cluster centers on a chart.

    Parameters:
    - centers: numpy array of cluster centers with shape (M, 2), where M is the number of centers,
      and each center is defined by (x_center, y_center).
    """
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


def plot_cluster_centers_with_bbox_centers(centers, bounding_boxes):
    """
    Plot the cluster centers and the centers of the bounding boxes on a chart.
    """

    # Convert lists to numpy arrays if they are not already
    if isinstance(bounding_boxes, list):
        bounding_boxes = np.array(bounding_boxes)
    if isinstance(centers, list):
        centers = np.array(centers)

    # Calculate the centers of the bounding boxes
    bbox_centers = np.c_[
        (bounding_boxes[:, 0] + bounding_boxes[:, 2]) / 2,  # x_center
        (bounding_boxes[:, 1] + bounding_boxes[:, 3]) / 2  # y_center
    ]

    # Plot the centers of the bounding boxes
    plt.scatter(bbox_centers[:, 0], bbox_centers[:, 1], c='blue', marker='o', label='Bounding Box Centers')

    # Plot the cluster centers
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
