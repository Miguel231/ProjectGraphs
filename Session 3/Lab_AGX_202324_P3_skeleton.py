import networkx as nx

# ------- IMPLEMENT HERE ANY AUXILIARY FUNCTIONS NEEDED ------- #


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
    pass
    # ------------------- END OF MAIN ------------------------ #
