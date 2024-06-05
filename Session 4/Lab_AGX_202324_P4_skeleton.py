import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import sys
sys.path.insert(0, 'Session 3')
import Lab_AGX_202324_P3_skeleton as Lab3

sys.path.insert(0, 'Session 2')
import Lab_AGX_202324_P2_skeleton as Lab2
# ------- IMPLEMENT HERE ANY AUXILIARY FUNCTIONS NEEDED ------- #
def get_degree_distribution(graph):
    """
    Generate a degree distribution dictionary from a graph.
    
    :param graph: NetworkX graph object
    :return: Dictionary with degree counts
    """
    degree_dict = {}
    for degree in dict(nx.degree(graph)).values():
        if degree in degree_dict:
            degree_dict[degree] += 1
        else:
            degree_dict[degree] = 1
    return degree_dict
# --------------- END OF AUXILIARY FUNCTIONS ------------------ #

def plot_degree_distribution(degree_dict: dict, normalized: bool = False, loglog: bool = False) -> None:
    """
    Plot degree distribution from dictionary of degree counts.

    :param degree_dict: dictionary of degree counts (keys are degrees, values are occurrences).
    :param normalized: boolean indicating whether to plot absolute counts or probabilities.
    :param loglog: boolean indicating whether to plot in log-log scale.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    degrees = list(degree_dict.keys())
    counts = list(degree_dict.values())
    
    if normalized:
        total = sum(counts)
        counts = [x / total for x in counts]
    
    plt.figure(figsize=(15, 7))  
    plt.bar(degrees, counts, width=0.8)
    
    if loglog:
        plt.xscale('log')
        plt.yscale('log')
    
    plt.title('Degree Distribution')
    plt.xlabel('Degree')
    plt.ylabel('Probability' if normalized else 'Count')
    plt.grid(True)
    plt.show()
    # ----------------- END OF FUNCTION --------------------- #


def plot_audio_features(artists_audio_feat: pd.DataFrame, artist1_id: str, artist2_id: str) -> None:
    """
    Plot a (single) figure with a plot of mean audio features of two different artists.

    :param artists_audio_feat: dataframe with mean audio features of artists.
    :param artist1_id: string with id of artist 1.
    :param artist2_id: string with id of artist 2.
    :return: None
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    feat1 = artists_audio_feat[artists_audio_feat['Artist Name'] == artist1_id]
    feat2 = artists_audio_feat[artists_audio_feat['Artist Name'] == artist2_id]
    
    ind = np.arange(len(feat1.columns) - 1)  #skip 'Artist Name'
    width = 0.35
    
    fig, ax = plt.subplots()
    bars1 = ax.bar(ind - width/2, feat1.iloc[0, 1:], width, label=feat1['Artist Name'].iloc[0])
    bars2 = ax.bar(ind + width/2, feat2.iloc[0, 1:], width, label=feat2['Artist Name'].iloc[0])
    
    ax.set_ylabel('Feature Value')
    ax.set_title('Audio Features Comparison')
    ax.set_xticks(ind)
    ax.set_xticklabels(feat1.columns[1:], rotation=90)
    ax.legend()
    
    plt.show()
    # ----------------- END OF FUNCTION --------------------- #


def plot_similarity_heatmap(artist_audio_features_df: pd.DataFrame, similarity: str, out_filename: str = None) -> None:
    """
    Plot a heatmap of the similarity between artists.

    :param artist_audio_features_df: dataframe with mean audio features of artists.
    :param similarity: string with similarity measure to use.
    :param out_filename: name of the file to save the plot. If None, the plot is not saved.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    features = artist_audio_features_df.drop(['Artist Name'], axis=1).values
    artists = artist_audio_features_df['Artist Name'].values
    
    if similarity == 'cosine':
        sim_matrix = cosine_similarity(features)
    elif similarity == 'euclidean':
        sim_matrix = euclidean_distances(features)
        sim_matrix = 1 - (sim_matrix / np.max(sim_matrix))  # Normalize distances to [0, 1] range

    # Increase figure size dynamically based on the number of artists
    fig_size = max(10, len(artists) * 0.3)
    plt.figure(figsize=(fig_size, fig_size))
    
    sns.heatmap(sim_matrix, annot=True, cmap='coolwarm', 
                xticklabels=artist_audio_features_df['Artist Name'], 
                yticklabels=artist_audio_features_df['Artist Name'])
    
    plt.title('Artist Similarity Heatmap')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)  # Ensure y labels are horizontal for readability

    if out_filename:
        plt.savefig(out_filename)
    #plt.show()
    # ----------------- END OF FUNCTION --------------------- #

def plot_graph_components(g: nx.Graph):
    """
    Plot the graph with its components highlighted.
    
    :param g: NetworkX graph
    """
    # Set the figure size
    plt.figure(figsize=(12, 8))
    
    # Generate layout for the graph
    pos = nx.spring_layout(g)
    
    # Draw nodes and edges
    nx.draw_networkx_edges(g, pos, alpha=0.3)
    nx.draw_networkx_nodes(g, pos, node_size=20, cmap=plt.cm.jet)
    
    # Draw labels for the largest connected components
    largest_components = sorted(nx.connected_components(g), key=len, reverse=True)
    if len(largest_components) > 0:
        largest_component = largest_components[0]
        nx.draw_networkx_labels(g.subgraph(largest_component), pos, labels={n: str(n) for n in largest_component}, font_size=12, font_color='red')

    plt.title('Graph Visualization with Highlighted Largest Component')
    plt.axis('off')  # Turn off the axis
    plt.show()


if __name__ == "__main__":
    # ------- IMPLEMENT HERE THE MAIN FOR THIS SESSION ------- #
    gB_p = nx.read_graphml('Session 2/gBp.graphml')
    gD_p = nx.read_graphml('Session 2/gDp.graphml')
    gw = nx.read_graphml('Session 2/gw.graphml')
    songs_mean = pd.read_csv('Session 2/songs_mean.csv')

    #Part a)
    gB_p_degree_dict = Lab3.get_degree_distribution(gB_p)
    #plot_degree_distribution(gB_p_degree_dict, normalized=True, loglog=False)

    gD_p_degree_dict = Lab3.get_degree_distribution(gD_p)
    #plot_degree_distribution(gB_p_degree_dict, normalized=True, loglog=False)
    
    gw_degree_dict = Lab3.get_degree_distribution(gw)
    #plot_degree_distribution(gB_p_degree_dict, normalized=True, loglog=False)
    
    #Part b)
    #similarity_scores = {node: nx.get_node_attributes(gw, 'weight')[node] for node in gw.nodes}
    #most_similar_artist = max(similarity_scores, key=similarity_scores.get)
    #least_similar_artist = min(similarity_scores, key=similarity_scores.get)
    
    #plot_audio_features(songs_mean, 'Taylor Swift ID', most_similar_artist)
    
    #Part c)
    #plot_audio_features(songs_mean, 'Taylor Swift ID', least_similar_artist)
    
    #Part d)
    #plot_similarity_heatmap(songs_mean, similarity='cosine', out_filename = 'Session 4/heatmap.png')
    #plot_similarity_heatmap(songs_mean, similarity='euclidean', out_filename = 'Session 4/heatmap2.png')
    
    #Part e)  
    pruned_gw = Lab2.prune_low_weight_edges(gw, min_weight=0.1)  # Adjust `min_weight` as needed
    #no saleee
    
    
    # ------------------- END OF MAIN ------------------------ #
