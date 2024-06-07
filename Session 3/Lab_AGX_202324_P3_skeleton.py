import networkx as nx
import pandas as pd
import community.community_louvain as community_louvain

# ------- IMPLEMENT HERE ANY AUXILIARY FUNCTIONS NEEDED ------- #


def max_clique_size_with_min_2_cliques(g):
    """
    Determine the maximum clique size that generates at least 2 cliques in the graph g.

    :param g: networkx graph.
    :return: max clique size and the corresponding cliques.
    """
    max_clique = 2
    
    # Find all cliques in the graph
    cliques = list(nx.find_cliques(g))
    
    # Length of the maximal clique in g
    max_lenght_clique= len(max(cliques, key=len))
    
    # Iterate clique sizes from 2 to max_lenght_clique
    for size in range(2, max_lenght_clique + 1):

        # Filter cliques by the current size
        filtered_cliques = []
        for clique in cliques:
            if len(clique) >= size:
                filtered_cliques.append(clique)
        
        # Check if there are min 2 cliques oof the given size
        if len(filtered_cliques) >= 2:
            # Update the max_clique size 
            max_clique = size
    
    # Return the maximum clique size that generates at least 2 cliques
    return max_clique

# Function to get artist names from node IDs
def get_artist_names(graph, node_ids):
    artist_names = []
    for node_id in node_ids:
        if 'name' in graph.nodes[node_id]:
            artist_names.append(graph.nodes[node_id]['name'])
    return artist_names


# --------------- END OF AUXILIARY FUNCTIONS ------------------ #

def num_common_nodes(*arg):
    """
    Return the number of common nodes between a set of graphs.

    :param arg: (an undetermined number of) networkx graphs.
    :return: an integer, number of common nodes.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #

    # Start with set of nodes in the first graph
    first_graph = set(arg[0].nodes)
    
    # Intersect all the graphs passed to the function
    for g in arg[1:]:
        first_graph.intersection_update(g.nodes)
    
     # Return 'count' of intersected nodes with len
    return len(first_graph)
    # ----------------- END OF FUNCTION --------------------- #


def get_degree_distribution(g: nx.Graph) -> dict:
    """
    Get the degree distribution of the graph.

    :param g: networkx graph.
    :return: dictionary with degree distribution (keys are degrees, values are number of occurrences).
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    
    degree_distribution = {}

    # Iterate over the degrees of each node in the graph
    for nodes, degree in g.degree():

        # Check if degree is in the dictionary
        if degree not in degree_distribution:

            # Add the degree as a key and initialize with 0
            degree_distribution[degree] = 0

        # Update the count for the degree +1 
        degree_distribution[degree] += 1
    
    return degree_distribution

    # ----------------- END OF FUNCTION --------------------- #


def get_k_most_central(g: nx.Graph, metric: str, num_nodes: int) -> list:
    """
    Get the k most central nodes in the graph.

    :param g: networkx graph.
    :param metric: centrality metric. Can be (at least) 'degree', 'betweenness', 'closeness' or 'eigenvector'.
    :param num_nodes: number of nodes to return.
    :return: list with the top num_nodes nodes.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #

     # Calculate the centrality depending on the specified metric
    if metric == 'degree':
        centrality_measure = nx.degree_centrality(g)
    elif metric == 'betweenness':
        centrality_measure = nx.betweenness_centrality(g)
    elif metric == 'closeness':
        centrality_measure = nx.closeness_centrality(g)
    elif metric == 'eigenvector':
        centrality_measure = nx.eigenvector_centrality(g)
    else:
        raise ValueError("Invalid centrality metric specified.")

    # Sort in descending order through the items of the desired centrality_measure
    sort_nodes = sorted(centrality_measure.items(), key=lambda item: item[1], reverse=True) # reverse set to true to do descending order

    # Take the desired top nodes specified in the parameters of the function
    top_nodes = []
    for node, _ in sort_nodes[:num_nodes]:
        top_nodes.append(node)
    
    return top_nodes
    
    # ----------------- END OF FUNCTION --------------------- #


def find_cliques(g: nx.Graph, min_size_clique: int) -> tuple:
    """
    Find cliques in the graph g with size at least min_size_clique.

    :param g: networkx graph.
    :param min_size_clique: minimum size of the cliques to find.
    :return: two-element tuple, list of cliques (each clique is a list of nodes) and
        list of nodes in any of the cliques.
    """
    # Find all maximal cliques in the graph
    cliques = list(nx.find_cliques(g))
    
    # Filter cliques by minimum size returned by max_clique_size_with_min_2_cliques function
    filtered_cliques = []
    for clique in cliques:
        if len(clique) >= min_size_clique:
            filtered_cliques.append(clique)
    
    # Get nodes that are part of all the filtered cliques in a set
    nodes_in_cliques = set()
    for clique in filtered_cliques:
        for node in clique:
            nodes_in_cliques.add(node)
    
    return filtered_cliques, list(nodes_in_cliques)



def detect_communities(g: nx.Graph, method: str) -> tuple:
    """
    Detect communities in the graph g using the specified method.

    :param g: a networkx graph.
    :param method: string with the name of the method to use. Can be (at least) 'girvan-newman' or 'louvain'.
    :return: two-element tuple, list of communities (each community is a list of nodes) and modularity of the partition.
    """
    if method == 'girvan-newman':
        # Generate communities using the Girvan-Newman algorithm
        communities_generator = nx.algorithms.community.girvan_newman(g)
        
        # Find communities with size <= 10
        communities = None
        for communities in communities_generator:
            if len(communities) > 10:
                break
        
        # Convert to nested list
        communities = [list(c) for c in communities]
        
        # Calculate modularity
        modularity = nx.algorithms.community.modularity(g, communities)

    elif method == 'louvain':
        if nx.is_directed(g):
            g = g.to_undirected()
        
        # Generate communities using the Louvain algorithm
        partition = community_louvain.best_partition(g)
        communities = {}
        
        for node, community_id in partition.items():
            if community_id not in communities:
                communities[community_id] = []
            communities[community_id].append(node)
        
        # Convert communities to list of lists
        communities = list(communities.values())
        
        # Calculate modularity
        modularity = community_louvain.modularity(partition, g)

    else:
        raise ValueError("Method not supported. Use 'girvan-newman' or 'louvain'.")

    return communities, modularity



if __name__ == '__main__':
    # ------- IMPLEMENT HERE THE MAIN FOR THIS SESSION ------- #

    #___________________EXERCICE 1_________________________________________________________________________________________
    
    print('EXERCICE 1 \n')
    
    gB = nx.read_graphml('Session 1/gB.graphml')
    gD = nx.read_graphml('Session 1/gD.graphml')

    #a)
    common_nodes = num_common_nodes(gB,gD)
    print(f'Exe1 a) {common_nodes}')

    #b)
    gBp = nx.read_graphml('Session 2/gBp.graphml')
    common_nodes = num_common_nodes(gB,gBp)
    print(f'Exe1 b) {common_nodes}\n')

    #___________________EXERCICE 2_________________________________________________________________________________________
    print('________________________________________________________________')
    print('EXERCICE 2 \n')
    
    degree_centrality =get_k_most_central(g= gBp, metric ='degree', num_nodes = 25)
    betweenness_centrality =get_k_most_central(g= gBp, metric ='betweenness', num_nodes = 25)

    #print(f'Exe 2: Top_n Degree Centrality nodes {degree_centrality}; Top_n Betweenness Centrality nodes {betweenness_centrality}')


    common_nodes = len(set(degree_centrality).intersection(set(betweenness_centrality)))

    print(f'Common nodes:{common_nodes}\n')

    #___________________EXERCICE 3_________________________________________________________________________________________
    print('________________________________________________________________')
    print('EXERCISE 3 \n')
    
    gDp = nx.read_graphml('Session 2/gDp.graphml')

    # Determine max clique size for gBp and gDp
    max_clique_size_gBp = max_clique_size_with_min_2_cliques(gBp)
    max_clique_size_gDp = max_clique_size_with_min_2_cliques(gDp)

    # Find cliques of size greater than or equal to the determined min_size_clique
    cliques_gBp, nodes_in_cliques_gBp = find_cliques(gBp, max_clique_size_gBp)
    cliques_gDp, nodes_in_cliques_gDp = find_cliques(gDp, max_clique_size_gDp)

    print(f"Max clique size for gBp that generates at least 2 cliques: {max_clique_size_gBp}")
    print(f"Total number of cliques in gBp: {len(cliques_gBp)}")
    print(f"Total number of different nodes in all cliques in gBp: {len(nodes_in_cliques_gBp)}\n")

    print(f"Max clique size for gDp that generates at least 2 cliques: {max_clique_size_gDp}")
    print(f"Total number of cliques in gDp: {len(cliques_gDp)}")
    print(f"Total number of different nodes in all cliques in gDp: {len(nodes_in_cliques_gDp)}\n")

    # Compare results
    common_nodes_in_cliques = set(nodes_in_cliques_gBp).intersection(set(nodes_in_cliques_gDp))
    print(f"Total number of common nodes in all cliques in gBp and gDp: {len(common_nodes_in_cliques)}")

    #___________________EXERCICE 4_________________________________________________________________________________________
    print('________________________________________________________________')
    print('EXERCISE 4 \n')

    artist_info = pd.read_csv('Session 2/artist_mean.csv')
    artists_in_clique = artist_info[artist_info['Artist ID'].isin(cliques_gBp[1])]

    #print(artists_in_clique[['Artist Name','Danceability', 'Energy', 'Loudness', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo']])
    print(artists_in_clique[['Artist Name', 'Energy', 'Loudness', 'Valence', 'Tempo']])
    #___________________EXERCICE 5_________________________________________________________________________________________
    print('________________________________________________________________')
    print('EXERCISE 5 \n')

    # Detect communities using Louvain method
    communities_louvain, modularity_louvain = detect_communities(gD, 'louvain')
    print(f"Louvain Method: Modularity = {modularity_louvain}")
    print(f"Number of communities detected: {len(communities_louvain)}")

    #___________________EXERCICE 6_________________________________________________________________________________________
    print('________________________________________________________________')
    print('EXERCISE 6 \n')

    # a)

    sccs = list(nx.strongly_connected_components(gD))
    min_cost = len(sccs) * 100  # 100 euros per artist
    print(f'a) {min_cost}')

    # b)
    print('b) \n')
    print('gD)')
    budget = 400
    cost_per_artist = 100
    num_artists = budget // cost_per_artist

    betweenness_centrality_gD = get_k_most_central(g= gD, metric ='betweenness', num_nodes = num_artists)
    betweenness_centrality_gB = get_k_most_central(g= gB, metric ='betweenness', num_nodes = num_artists)

    artist_names_gD = get_artist_names(gD, betweenness_centrality_gD)
    artist_names_gB = get_artist_names(gB, betweenness_centrality_gB)

    for i in range(len(betweenness_centrality_gD)):
        node_id = betweenness_centrality_gD[i]
        artist_name = artist_names_gD[i]
        print(f"Artist ID: {node_id}, Artist Name: {artist_name}")



    print('gB)')
    for i in range(len(betweenness_centrality_gB)):
        node_id = betweenness_centrality_gB[i]
        artist_name = artist_names_gB[i]
        print(f"Artist ID: {node_id}, Artist Name: {artist_name}")

    
    #___________________EXERCICE 7_________________________________________________________________________________________
    print('________________________________________________________________')
    print('EXERCISE 7 \n')

        
    # Define the start and target artists
    start_artist = 'Taylor Swift'
    target_artist = 'THE DRIVER ERA'

    # Find the node IDs for the start and target artists
    start_node = None
    target_node = None
    for node in gB.nodes:
        if gB.nodes[node]['name'] == start_artist:
            start_node = node
        if gB.nodes[node]['name'] == target_artist:
            target_node = node

    # Ensure both artists are in the graph
    if start_node is None or target_node is None:
        print("One or both artists are not in the graph.")
    else:
        # Use BFS to find the shortest path
        try:
            shortest_path = nx.shortest_path(gB, source=start_node, target=target_node)
            shortest_path_names = [gB.nodes[node]['name'] for node in shortest_path]
            print(f"Minimum number of hops: {len(shortest_path) - 1}")
            print("Artists in the path:")
            for artist in shortest_path_names:
                print(artist)
        except nx.NetworkXNoPath:
            print(f"No path found from {start_artist} to {target_artist}.")

