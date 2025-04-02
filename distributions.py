import numpy as np

def generate_uniform_distribution(num_cities, x_range, y_range):
    x_coords = np.random.uniform(x_range[0], x_range[1], num_cities)
    y_coords = np.random.uniform(y_range[0], y_range[1], num_cities)
    return np.column_stack((x_coords, y_coords))

def generate_normal_distribution(num_cities, mean, std_dev):
    x_coords = np.random.normal(mean[0], std_dev[0], num_cities)
    y_coords = np.random.normal(mean[1], std_dev[1], num_cities)
    return np.column_stack((x_coords, y_coords))

def generate_clustered_distribution(num_cities, num_clusters, cluster_radius):
    clusters = []
    for _ in range(num_clusters):
        cluster_center = np.random.uniform(0, 100, 2)
        for _ in range(num_cities // num_clusters):
            x, y = np.random.normal(cluster_center, cluster_radius, 2)
            clusters.append([x, y])
    return np.array(clusters)

def generate_exponential_distribution(num_cities, scale):
    x_coords = np.random.exponential(scale, num_cities)
    y_coords = np.random.exponential(scale, num_cities)
    return np.column_stack((x_coords, y_coords))

def generate_poisson_distribution(num_cities, lam):
    x_coords = np.random.poisson(lam, num_cities)
    y_coords = np.random.poisson(lam, num_cities)
    return np.column_stack((x_coords, y_coords))