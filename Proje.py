import pandas as pd
import numpy as np
import networkx as nx
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from kneed import KneeLocator
from heapq import heappop, heappush
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
matplotlib.use('TkAgg')  # GUI penceresinde animasyon için
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('QtAgg') 

# Ensure compatibility with Spyder for animations
 # Use QtAgg for Spyder compatibility

# Step 1: Load the distance matrix from Excel
file_path = "C:\\Users\\NFS\\Desktop\\Yapay\\YapayOdev\\İhtiyacNoktaları.xlsx"
mesafe_matrisi_data = pd.read_excel(file_path, index_col=0)

# Ensure the matrix is properly formatted
mesafe_matrisi_data.index = mesafe_matrisi_data.index.str.strip()
mesafe_matrisi_data.columns = mesafe_matrisi_data.columns.str.strip()

# Convert the distance matrix to 2D coordinates using MDS
distance_array = mesafe_matrisi_data.values
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
coords = mds.fit_transform(distance_array)

# Apply the Elbow Method to find the optimal number of clusters
wcss = []
cluster_range = range(1, 11)

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(coords)
    wcss.append(kmeans.inertia_)

# Determine the optimal number of clusters
knee_locator = KneeLocator(cluster_range, wcss, curve="convex", direction="decreasing")
optimal_clusters = knee_locator.knee

# Apply K-Means with the optimal number of clusters
kmeans_optimal = KMeans(n_clusters=optimal_clusters, random_state=42)
cluster_labels = kmeans_optimal.fit_predict(coords)

# Add the cluster labels to the original DataFrame
mesafe_matrisi_data['Cluster'] = cluster_labels

# Assign geographical names to clusters based on centroids
centroids = kmeans_optimal.cluster_centers_
cluster_names = []

for x, y in centroids:
    if y > 0 and abs(y) > abs(x):  # North
        cluster_names.append("North")
    elif y < 0 and abs(y) > abs(x):  # South
        cluster_names.append("South")
    elif x > 0 and abs(x) > abs(y):  # East
        cluster_names.append("East")
    elif x < 0 and abs(x) > abs(y):  # West
        cluster_names.append("West")
        

# Map cluster names to cluster labels
cluster_label_to_name = {i: name for i, name in enumerate(cluster_names)}



# Create a figure for the plot
plt.figure(figsize=(12, 8))

# Scatter plot for each cluster
for cluster_label in np.unique(cluster_labels):
    # Get the points corresponding to the current cluster
    cluster_points = coords[cluster_labels == cluster_label]
    
    # Plot the cluster points with a unique color
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster_label}")

# Plot centroids and add names
for i, (x, y) in enumerate(centroids):
    # Mark centroids with a red star
    plt.scatter(x, y, c='red', s=200, marker='*')
    
    # Add the cluster name near the centroid
    cluster_name = cluster_label_to_name.get(i, f"Cluster {i}")  # Fetch cluster name or fallback to cluster ID
    plt.text(x, y, cluster_name, fontsize=12, fontweight='bold', ha='center', va='center', color='black')

# Optionally add the depot if needed
plt.scatter(0, 0, c='orange', s=300, marker='D', label='Depot')

# Add title, labels, legend, and grid
plt.title("Clusters Visualization with Centroids and Names")
plt.xlabel("MDS Dimension 1")
plt.ylabel("MDS Dimension 2")
plt.legend()
plt.grid()
plt.show()



# Map cluster names to cluster labels
cluster_label_to_name = {i: name for i, name in enumerate(cluster_names)}
mesafe_matrisi_data['Cluster_Name'] = mesafe_matrisi_data['Cluster'].map(cluster_label_to_name)

# Filter for West cluster
west_cluster_points = mesafe_matrisi_data[mesafe_matrisi_data['Cluster_Name'] == "West"].index.tolist()
distance_matrix = mesafe_matrisi_data.loc[west_cluster_points, west_cluster_points]


# Plot the West cluster points with names
fig, ax = plt.subplots(figsize=(10, 8))

# Scatter plot of West cluster points
x_coords = coords[[distance_matrix.index.get_loc(idx) for idx in west_cluster_points], 0]
y_coords = coords[[distance_matrix.index.get_loc(idx) for idx in west_cluster_points], 1]

ax.scatter(x_coords, y_coords, c='blue', label='West Locations')



# Add depot
ax.scatter(0, 0, c='red', label='Depot', s=100)

# Annotate each point with its name
for idx, (x, y) in zip(west_cluster_points, zip(x_coords, y_coords)):
    ax.text(x, y, idx, fontsize=10, ha='right', color='black')

# Add legend and labels
ax.legend()
ax.set_title('West Cluster Points with Names')
ax.set_xlabel('MDS Dimension 1')
ax.set_ylabel('MDS Dimension 2')

plt.grid(True)
plt.show()


# Ensure correct indexing for distance matrix
distance_matrix.index = distance_matrix.index.str.strip()
distance_matrix.columns = distance_matrix.columns.str.strip()

# Filter needs_data for West cluster
needs_data = pd.read_excel("C:\\Users\\NFS\\Desktop\\YapayOdev\\Yardım_Talep_Düzeltildi.xlsx")
needs_data['WestihtiyacNoktasi'] = needs_data['WestihtiyacNoktasi'].str.strip()
needs_data = needs_data[needs_data['WestihtiyacNoktasi'].isin(west_cluster_points)]
remaining_needs = needs_data.set_index('WestihtiyacNoktasi')['ihtiyacSayisi'].to_dict()

# Parameters
DRONE_CAPACITY = 30
DEPO_LOCATION = "Depo_West"  # Assuming there is a West-specific depot

# Convert distance matrix to graph
def create_graph(matrix):
    G = nx.Graph()
    for i, point_a in enumerate(matrix.index):
        for j, point_b in enumerate(matrix.columns):
            if i != j:
                G.add_edge(point_a, point_b, weight=matrix.loc[point_a, point_b])
    return G

graph = create_graph(distance_matrix)

# Add depot to the graph
graph.add_node(DEPO_LOCATION)
for node in distance_matrix.index:
    if node != DEPO_LOCATION:
        graph.add_edge(DEPO_LOCATION, node, weight=0)  # Assuming 0 distance for initialization

# A* Algorithm for shortest path
def a_star_search(graph, start, goal):
    frontier = []
    heappush(frontier, (0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}

    while frontier:
        current_priority, current_node = heappop(frontier)

        if current_node == goal:
            break

        for neighbor in graph.neighbors(current_node):
            new_cost = cost_so_far[current_node] + graph[current_node][neighbor]['weight']
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost
                heappush(frontier, (priority, neighbor))
                came_from[neighbor] = current_node

    return came_from, cost_so_far

delivery_route = []
current_location = DEPO_LOCATION

def get_next_target(remaining):
    return next((loc for loc, need in remaining.items() if need > 0), None)

while any(remaining_needs.values()):
    next_target = get_next_target(remaining_needs)
    if not next_target:
        break

    # Compute path to next target
    path, cost = a_star_search(graph, current_location, next_target)
    delivery_route.append((current_location, next_target, cost[next_target]))

    # Check drone capacity
    if remaining_needs[next_target] <= DRONE_CAPACITY:
        DRONE_CAPACITY -= remaining_needs[next_target]
        remaining_needs[next_target] = 0
    else:
        remaining_needs[next_target] -= DRONE_CAPACITY
        DRONE_CAPACITY = 0

    # Return to depot if drone is empty
    if DRONE_CAPACITY == 0:
        return_path, return_cost = a_star_search(graph, next_target, DEPO_LOCATION)
        delivery_route.append((next_target, DEPO_LOCATION, return_cost[DEPO_LOCATION]))
        DRONE_CAPACITY = 30  # Reload drone
        current_location = DEPO_LOCATION
    else:
        current_location = next_target

# Print delivery route
for route in delivery_route:
    print(f"Drone moved from {route[0]} to {route[1]} (Cost: {route[2]} units)")

# Save the route to Excel
output = pd.DataFrame(delivery_route, columns=['Start', 'End', 'Cost'])
output.to_excel("Drone_Delivery_Route.xlsx", index=False)

# Create animation of the delivery route
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(coords[[distance_matrix.index.get_loc(idx) for idx in west_cluster_points], 0], coords[[distance_matrix.index.get_loc(idx) for idx in west_cluster_points], 1], c='blue', label='West Locations')
ax.scatter(0, 0, c='red', label='Depot', s=100)  # Assume depot at (0, 0)
ax.legend()

# Extract coordinates of the route
location_map = {name: coords[distance_matrix.index.get_loc(name)] for name in west_cluster_points if name in distance_matrix.index}
location_map[DEPO_LOCATION] = [0, 0]  # Add depot coordinates

route_coords = [(location_map[start], location_map[end]) for start, end, _ in delivery_route]

# Debugging: Ensure route_coords is correct
print("Route Coordinates:", route_coords)

def update(frame):
    ax.clear()
    ax.scatter(coords[[distance_matrix.index.get_loc(idx) for idx in west_cluster_points], 0], coords[[distance_matrix.index.get_loc(idx) for idx in west_cluster_points], 1], c='blue', label='West Locations')
    ax.scatter(0, 0, c='red', label='Depot', s=100)
    ax.legend()

    # Smooth movement between points
    if frame == 0:
        return

    start, end = route_coords[min(frame - 1, len(route_coords) - 1)]
    num_steps = 50  # Steps for smooth movement
    t = (frame % num_steps) / num_steps  # Normalized time
    current_pos = np.array(start) * (1 - t) + np.array(end) * t
    
    # Add names to points
    for idx in west_cluster_points:
        x, y = coords[distance_matrix.index.get_loc(idx)]
        ax.text(x, y, idx, fontsize=8, ha='right', color='black')  # Label each point


    # Plot drone current position
    ax.plot([start[0], current_pos[0]], [start[1], current_pos[1]], 'k-', lw=2)
    ax.scatter(current_pos[0], current_pos[1], c='green', s=100, label='Drone')

ani = FuncAnimation(fig, update, frames=len(route_coords) + 1, interval=1000, repeat=False)
plt.show()







