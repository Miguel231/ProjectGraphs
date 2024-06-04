import networkx as nx

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
    
    # lenght of the maximal clique in g
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
    :param method: string with the name of the method to use. Can be (at least) 'givarn-newman' or 'louvain'.
    :return: two-element tuple, list of communities (each community is a list of nodes) and modularity of the partition.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    pass
    # ----------------- END OF FUNCTION --------------------- #


if __name__ == '__main__':
    # ------- IMPLEMENT HERE THE MAIN FOR THIS SESSION ------- #

    #EXERCICE 1
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

    #EXERCICE 2

    print('EXERCICE 2 \n')
    degree_centrality =get_k_most_central(g= gBp, metric ='degree', num_nodes = 25)
    betweenness_centrality =get_k_most_central(g= gBp, metric ='betweenness', num_nodes = 25)

    #print(f'Exe 2: Top_n Degree Centrality nodes {degree_centrality}; Top_n Betweenness Centrality nodes {betweenness_centrality}')


    common_nodes = len(set(degree_centrality).intersection(set(betweenness_centrality)))

    print(f'Common nodes:{common_nodes}\n')

    #EXERCICE 3
    print('EXERCICE 3 \n')

    gDp = nx.read_graphml('Session 2/gDp.graphml')

    clique_gbp = find_cliques(gBp, min_size_clique= max_clique_size_with_min_2_cliques(gBp))
    

    print(f'Total number of different nodes part of the cliques of gBp; {clique_gbp}')

    clique_gdp = find_cliques(gDp, min_size_clique= max_clique_size_with_min_2_cliques(gDp))

    print(f'Total number of different nodes part of the cliques of gBp; {clique_gdp}')




    





    # ------------------- END OF MAIN ------------------------ #
