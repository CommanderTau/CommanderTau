import numpy as np
import networkx as nx

graph_matrix = np.array([
    [0, 3, 4, 4, 0, 16],
    [3, 0, 0, 5, 0, 0],
    [4, 0, 0, 3, 0, 0],
    [4, 5, 3, 0, 6, 10],
    [0, 0, 0, 6, 0, 3],
    [16, 0, 0, 10, 3, 0]
])
G = nx.from_numpy_array(graph_matrix, create_using=nx.Graph)
start_node = 0
end_node = 5
try:
    shortest_path = nx.shortest_path(G, source=start_node, target=end_node, weight='weight')
    shortest_path_length = nx.shortest_path_length(G, source=start_node, target=end_node, weight='weight')

    print("Кратчайший путь:", " -> ".join(chr(65 + node) for node in shortest_path))
    print(f"Длина пути: {shortest_path_length}")
except nx.NetworkXNoPath:
    print("Нет пути между заданными точками.")