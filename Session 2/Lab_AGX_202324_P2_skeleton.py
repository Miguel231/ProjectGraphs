import networkx as nx
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import numpy as np
# ------- IMPLEMENT HERE ANY AUXILIARY FUNCTIONS NEEDED ------- #
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
        if g.has_edge(v, u):
            undirected_g.add_edge(u, v)
            added_edges += 1

    print(f"Total bidirectional edges added: {added_edges}")

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
    pruned_g = g.copy()
    low_degree_nodes = [node for node, degree in pruned_g.degree() if degree < min_degree]
    pruned_g.remove_nodes_from(low_degree_nodes)
    pruned_g.remove_nodes_from(list(nx.isolates(pruned_g)))
    
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

    if min_percentile is not None:
        weights = [data['weight'] for u, v, data in g.edges(data=True)]
        min_weight = np.percentile(weights, min_percentile)

    edges_to_remove = [(u, v) for u, v, data in g.edges(data=True) if data['weight'] < min_weight]
    g.remove_edges_from(edges_to_remove)
    g.remove_nodes_from(list(nx.isolates(g)))

    nx.write_graphml(g, out_filename)
    return g
    # ----------------- END OF FUNCTION --------------------- #


def compute_mean_audio_features(tracks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the mean audio features for tracks of the same artist.

    :param tracks_df: tracks dataframe (with audio features per each track).
    :return: artist dataframe (with mean audio features per each artist).
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    audio_characteristic = ['Danceability', 'Energy', 'Loudness', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo']
    artist = tracks_df.groupby(['Artist ID', 'Artist Name'])[audio_characteristic].mean().reset_index()
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

    artist_ids = artist_audio_features_df['Artist ID'].tolist()
    features = artist_audio_features_df.drop(columns=['Artist ID', 'Artist Name']).values

    if similarity == 'cosine':
        sim_matrix = cosine_similarity(features)
    elif similarity == 'euclidean':
        sim_matrix = euclidean_distances(features)
        sim_matrix = 1 - (sim_matrix / sim_matrix.max())

    for idx, artist_id in enumerate(artist_ids):
        graph.add_node(artist_id, name=artist_audio_features_df.iloc[idx]['Artist Name'])

    for i in range(len(artist_ids)):
        for j in range(i + 1, len(artist_ids)):  
            if sim_matrix[i, j] > 0:  
                graph.add_edge(artist_ids[i], artist_ids[j], weight=sim_matrix[i, j])

    nx.write_graphml(graph, out_filename)
    return graph
    # ----------------- END OF FUNCTION --------------------- #


if __name__ == "__main__":
    # ------- IMPLEMENT HERE THE MAIN FOR THIS SESSION ------- #
    gB = nx.read_graphml('Session 1/gB.graphml')
    gD = nx.read_graphml('Session 1/gD.graphml')
    songs = pd.read_csv('Session 1/songs.csv')

    #Part a)
    gBp = retrieve_bidirectional_edges(gB, "Session 2/gBp.graphml")
    gDp = retrieve_bidirectional_edges(gD, "Session 2/gDp.graphml")

    #Part b)
    artist_audio = compute_mean_audio_features(songs)
    gw = create_similarity_graph(artist_audio, similarity='cosine', out_filename="Session 2/gw.graphml")
    

    
    # ------------------- END OF MAIN ------------------------ #
