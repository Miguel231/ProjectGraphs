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
    all_cliques = list(nx.find_cliques(g))
    
    # Length of the maximal clique in g
    max_clique_length = len(max(all_cliques, key=len))
    
    # Iterate clique sizes from 2 to the size of the largest clique
    for size in range(2, max_clique_length + 1):
        # Filter cliques by the current size
        filtered_cliques = []
        for clique in all_cliques:
            if len(clique) >= size:
                filtered_cliques.append(clique)
        
        # Check if there are at least 2 cliques of the current size
        if len(filtered_cliques) >= 2:
            # Update the maximum clique size
            max_clique = size
    
    # Return the maximum clique size that generates at least 2 cliques
    return max_clique

def select_artists_for_ad_spread(g: nx.DiGraph, budget: int) -> list:
    """
    Select artists to maximize ad spread within the given budget.

    :param g: networkx directed graph.
    :param budget: total budget in euros.
    :return: list of selected artists.
    """
    cost_per_artist = 100
    num_artists = budget // cost_per_artist

    out_degree_centrality = nx.out_degree_centrality(g)
    sorted_artists = sorted(out_degree_centrality.items(), key=lambda item: item[1], reverse=True)

    selected_artists = [artist for artist, _ in sorted_artists[:num_artists]]
    return selected_artists

# --------------- END OF AUXILIARY FUNCTIONS ------------------ #

def num_common_nodes(*arg):
    """
    Return the number of common nodes between a set of graphs.

    :param arg: (an undetermined number of) networkx graphs.
    :return: an integer, number of common nodes.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    if not arg:
        return 0
    
    # Start with the node set of the first graph
    common_nodes = set(arg[0].nodes)
    
    # Intersect with the node sets of all other graphs
    for g in arg[1:]:
        common_nodes.intersection_update(g.nodes)
    
    return len(common_nodes)
    # ----------------- END OF FUNCTION --------------------- #


def get_degree_distribution(g: nx.Graph) -> dict:
    """
    Get the degree distribution of the graph.

    :param g: networkx graph.
    :return: dictionary with degree distribution (keys are degrees, values are number of occurrences).
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    
    degree_distribution = {}
    for _, degree in g.degree():
        if degree not in degree_distribution:
            degree_distribution[degree] = 0
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

    # Sort in descending order through the items of the centrality_measure output
    sorted_nodes = sorted(centrality_measure.items(), key=lambda item: item[1], reverse=True)

    # take the desired top nodes
    top_nodes = [node for node, _ in sorted_nodes[:num_nodes]]
    
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
    all_cliques = list(nx.find_cliques(g))
    
    # Filter cliques by minimum size
    filtered_cliques = [clique for clique in all_cliques if len(clique) >= min_size_clique]
    
    # Collect all nodes that are part of any filtered clique
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
        import itertools
        communities_generator = nx.algorithms.community.girvan_newman(g)
        limited = itertools.takewhile(lambda c: len(c) <= 10, communities_generator)
        communities = list(limited)[-1]
        communities = [list(c) for c in communities]
        modularity = nx.algorithms.community.modularity(g, communities)

    elif method == 'louvain':
        if nx.is_directed(g):
            g = g.to_undirected()
        partition = community_louvain.best_partition(g)
        communities = {}
        for node, community_id in partition.items():
            if community_id not in communities:
                communities[community_id] = []
            communities[community_id].append(node)
        communities = list(communities.values())
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

    print('EXERCICE 2 \n')
    degree_centrality =get_k_most_central(g= gBp, metric ='degree', num_nodes = 25)
    betweenness_centrality =get_k_most_central(g= gBp, metric ='betweenness', num_nodes = 25)

    #print(f'Exe 2: Top_n Degree Centrality nodes {degree_centrality}; Top_n Betweenness Centrality nodes {betweenness_centrality}')


    common_nodes = len(set(degree_centrality).intersection(set(betweenness_centrality)))

    print(f'Common nodes:{common_nodes}\n')

    #___________________EXERCICE 3_________________________________________________________________________________________

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

    print('EXERCISE 4 \n')

    pd.set_option('display.max_rows', None)

    artist_info = pd.read_csv('Session 1/songs.csv')
    artists_in_clique = artist_info[artist_info['Artist ID'].isin(cliques_gBp[1])]

    print(artists_in_clique[['Danceability', 'Energy', 'Loudness', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo']])

    #___________________EXERCICE 5_________________________________________________________________________________________

    print('EXERCISE 5 \n')

     # Detect communities using Louvain method
    communities_louvain, modularity_louvain = detect_communities(gD, 'louvain')
    print(f"Louvain Method: Modularity = {modularity_louvain}")
    print(f"Number of communities detected: {len(communities_louvain)}")

    #___________________EXERCICE 6_________________________________________________________________________________________

    print('EXERCISE 6 \n')

    # a)

    sccs = list(nx.strongly_connected_components(gD))
    min_cost = len(sccs) * 100  # 100 euros per artist
    print(f'a: {min_cost}')

    '''
    To ensure that a user who listens to music infinitely will hear your ad at some point, you need 
    to place ads on a set of artists such that every strongly connected component (SCC) of the graph is covered.

    '''
    # b)

    selected_artists_gD = select_artists_for_ad_spread(gD, 400)
    print(selected_artists_gD)
    '''
    With a budget of 400 euros, you can select up to 4 artists (100 euros per artist). The goal is to
    maximize the spread of your ad. One way to achieve this is by selecting artists with the highest out-degree centrality,
    as they can reach the most other artists.

    '''

     
    # ------------------- END OF MAIN ------------------------ #
