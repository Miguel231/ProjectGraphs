import networkx as nx
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import numpy as np

# ------- IMPLEMENT HERE ANY AUXILIARY FUNCTIONS NEEDED ------- #
def pair_similarity_artists(gw):
    """
    Finds the pairs of artists who are the most and the least similar based on the weight of the edges between them.

    :param gw: networkx graph where nodes represent artists and edges contain 'weight' attributes
    :return: A tuple containing:
             - The most similar artists with their similarity score.
             - The least similar artists with their similarity score.
    """
    max_weight = 0  #to ensure any weight is larger
    min_weight = 1  #to ensure any weight is smaller
    max_pair = None #to store the artist with the highest similarity
    min_pair = None #to store the artist with the lowest similarity

    #loop through all edges and find the max and min edges
    for u, v, data in gw.edges(data=True):
        weight = data['weight'] #extract the weight
        if weight > max_weight:
            max_weight = weight #update the max weight if it's greater
            max_pair = (u, v, weight) #update the artist pair
        if weight < min_weight:
            min_weight = weight #the same with the min weight
            min_pair = (u, v, weight)
    
    return max_pair, min_pair

def artist_similarity(gw):
    """
    Calculates the average similarity each artist has to all other artists in the graph
    and identifies the most and least similar artist.

    :param gw: networkx graph where nodes represent artists and edges contain 'weight' attributes
    :return: a tuple containing:
             - ID of the most similar artist and their score
             - ID of the least similar artist and their score
    """
    scores = {} #dict to store the avg scores for each artist
    #loop through each node in the graph
    for node in gw.nodes():
        connected_edges = gw.edges(node, data=True) #all edges for the node
        total_weight = sum(data['weight'] for _, _, data in connected_edges)
        #avg by dividing the total weight by the degree of this node
        scores[node] = total_weight / gw.degree(node)
    
    most_similar = max(scores, key=scores.get) #artist with the max value
    least_similar = min(scores, key=scores.get)#artist with the lowest value
    
    return most_similar, scores[most_similar], least_similar, scores[least_similar]



# --------------- END OF AUXILIARY FUNCTIONS ------------------ #

def retrieve_bidirectional_edges(g: nx.DiGraph, out_filename: str) -> nx.Graph:
    """
    Convert a directed graph into an undirected graph by considering bidirectional edges only.

    :param g: a networkx digraph.
    :param out_filename: name of the file that will be saved.
    :return: a networkx undirected graph.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    undirected_g = nx.Graph()

    added_edges = 0
    for u, v in g.edges():
        if g.has_edge(v, u): #if the opposite edge exists, makes it bidirectional
            undirected_g.add_edge(u, v)
            added_edges += 1

    print(f"Total bidirectional edges added: {added_edges}") #to check if it works

    nx.write_graphml(undirected_g, out_filename)
    return undirected_g
    # ----------------- END OF FUNCTION --------------------- #


def prune_low_degree_nodes(g: nx.Graph, min_degree: int, out_filename: str) -> nx.Graph:
    """
    Prune a graph by removing nodes with degree < min_degree.

    :param g: a networkx graph.
    :param min_degree: lower bound value for the degree.
    :param out_filename: name of the file that will be saved.
    :return: a pruned networkx graph.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    pruned_g = g.copy() #copy of the graph
    low_degree_nodes = [node for node, degree in pruned_g.degree() if degree < min_degree]
    pruned_g.remove_nodes_from(low_degree_nodes) #remove nodes with low degree
    pruned_g.remove_nodes_from(list(nx.isolates(pruned_g))) #remove isolated nodes
    
    if out_filename:
        nx.write_graphml(pruned_g, out_filename)

    return pruned_g
    # ----------------- END OF FUNCTION --------------------- #


def prune_low_weight_edges(g: nx.Graph, min_weight=None, min_percentile=None, out_filename: str = None) -> nx.Graph:
    """
    Prune a graph by removing edges with weight < threshold. Threshold can be specified as a value or as a percentile.

    :param g: a weighted networkx graph.
    :param min_weight: lower bound value for the weight.
    :param min_percentile: lower bound percentile for the weight.
    :param out_filename: name of the file that will be saved.
    :return: a pruned networkx graph.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    if min_weight is None and min_percentile is None:
        raise ValueError("min_weight or min_percentile must be specified")
    if min_weight is not None and min_percentile is not None:
        raise ValueError("Specify only one of min_weight or min_percentile")

    pruned_g = g.copy() #copy of hte graph

    if min_percentile is not None: # process the percentile value to obtain the min_weight
        weights = [data['weight'] for u, v, data in pruned_g.edges(data=True)]
        min_weight = np.percentile(weights, min_percentile)

    low_edges = [(u, v) for u, v, data in pruned_g.edges(data=True) if data['weight'] < min_weight]
    pruned_g.remove_edges_from(low_edges) #remove edges below the min_weight
    pruned_g.remove_nodes_from(list(nx.isolates(pruned_g))) #remove isolated nodes

    if out_filename:
        nx.write_graphml(pruned_g, out_filename)
        
    return pruned_g
    # ----------------- END OF FUNCTION --------------------- #


def compute_mean_audio_features(tracks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the mean audio features for tracks of the same artist.

    :param tracks_df: tracks dataframe (with audio features per each track).
    :return: artist dataframe (with mean audio features per each artist).
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    #all the audio features
    audio_characteristic = ['Danceability', 'Energy', 'Loudness', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo']
    
    #group by artist and compute mean for each feature
    artist = tracks_df.groupby(['Artist ID','Artist Name'])[audio_characteristic].mean().reset_index()
    return artist

    # ----------------- END OF FUNCTION --------------------- #


def create_similarity_graph(artist_audio_features_df: pd.DataFrame, similarity: str, out_filename: str = None) -> \
        nx.Graph:
    """
    Create a similarity graph from a dataframe with mean audio features per artist.

    :param artist_audio_features_df: dataframe with mean audio features per artist.
    :param similarity: the name of the similarity metric to use (e.g. "cosine" or "euclidean").
    :param out_filename: name of the file that will be saved.
    :return: a networkx graph with the similarity between artists as edge weights.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    graph = nx.Graph()
    
    #list of names of each artist
    artist_names = artist_audio_features_df['Artist Name'].tolist()
    #features values
    features = artist_audio_features_df.drop(columns=['Artist ID','Artist Name']).values

    #similarity matrix based on selected metric
    if similarity == 'cosine':
        sim_matrix = cosine_similarity(features)

    elif similarity == 'euclidean':
        sim_matrix = euclidean_distances(features)
        sim_matrix = 1 - (sim_matrix / sim_matrix.max()) #to normalize euclidean to similarity range

    #add nodes and weighted edges based
    for artist in artist_names:
        graph.add_node(artist, name=artist)

    for i in range(len(artist_names)):
        for j in range(i + 1, len(artist_names)):  
            graph.add_edge(artist_names[i], artist_names[j], weight=sim_matrix[i, j])

    if out_filename:
        nx.write_graphml(graph, out_filename)

    return graph
    # ----------------- END OF FUNCTION --------------------- #


if __name__ == "__main__":
    # ------- IMPLEMENT HERE THE MAIN FOR THIS SESSION ------- #
    #read the previous files
    gB = nx.read_graphml('Session 1/gB.graphml')
    gD = nx.read_graphml('Session 1/gD.graphml')
    songs = pd.read_csv('Session 1/songs.csv')

    #Part a)

    gBp = retrieve_bidirectional_edges(gB, "Session 2/gBp.graphml")
    gDp = retrieve_bidirectional_edges(gD, "Session 2/gDp.graphml")


    #Part b)
    artist_audio = compute_mean_audio_features(songs)
    artist_audio.to_csv('Session 2/artist_mean.csv', index=False)
    gw = create_similarity_graph(artist_audio, similarity='cosine', out_filename="Session 2/gw.graphml")
    
    #EXERCISE 1
    print(f'\nEXERCICE 1\n')
    num_wcc_gB = nx.number_weakly_connected_components(gB)
    num_wcc_gD = nx.number_weakly_connected_components(gD)

    # Check the number of strongly connected components
    num_scc_gB = nx.number_strongly_connected_components(gB)
    num_scc_gD = nx.number_strongly_connected_components(gD)

    print(f"Nº of weakly connected components in gB: {num_wcc_gB}")
    print(f"Nº of weakly connected components in gD: {num_wcc_gD}")
    print(f"Nº of strongly connected components in gB: {num_scc_gB}")
    print(f"Nº of strongly connected components in gD: {num_scc_gD}")

    #EXERCISE 2
    print(f'\nEXERCICE 2\n')

    num_connected_components_gBp = nx.number_connected_components(gBp)
    num_connected_components_gDp = nx.number_connected_components(gDp)

    print(f"Number of connected components in gBp: {num_connected_components_gBp}")
    print(f"Number of connected components in gDp: {num_connected_components_gDp}")

    #EXERCISE 3
    print(f'\nEXERCICE 3\n')
        #a)
    most_similar, least_similar = pair_similarity_artists(gw)
    print("Most similar artists:")
    print(f"{most_similar[0]} and {most_similar[1]}, score: {most_similar[2]}")
    print("\nLeast similar artists:")
    print(f"{least_similar[0]} and {least_similar[1]}, score: {least_similar[2]}")
        
        #b)
    most_similar_artist, most_score, least_similar_artist, least_score = artist_similarity(gw)
    print("\nArtist most similar to others:")
    print(f"{most_similar_artist} with an average score of {most_score}")
    print("\nArtist least similar to others:")
    print(f"{least_similar_artist} with an average score of {least_score}")
    # ------------------- END OF MAIN ------------------------ #
