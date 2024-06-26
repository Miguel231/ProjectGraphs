import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import sys

#import functions of different sessions
sys.path.insert(0, 'Session 3')
import Lab_AGX_202324_P3_skeleton as Lab3

sys.path.insert(0, 'Session 2')
import Lab_AGX_202324_P2_skeleton as Lab2
# ------- IMPLEMENT HERE ANY AUXILIARY FUNCTIONS NEEDED ------- #
def most_and_less_similar_artist(graph, artist_name):
    """
    Finds the most and least similar artists to the specified artist from gw.

    :param graph: NetworkX graph with artists as nodes and similarity as edge weights.
    :param artist_name: Name of the artist to find similarities for.
    :return: Names of the most and least similar artists.
    """
    neighbors = graph[artist_name] #access the neighboring nodes of the given artist

    most_similar_artist = None
    less_similar_artist = None
    max_similarity = 0  
    min_similarity = 1  

    #loop through all neighboring nodes to find the maximum and minimum similarities
    for artist, weight in neighbors.items():
        similarity = weight['weight']
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_artist = artist
        if similarity < min_similarity:
            min_similarity = similarity
            less_similar_artist = artist

    return most_similar_artist, less_similar_artist
# --------------- END OF AUXILIARY FUNCTIONS ------------------ #

def plot_degree_distribution(degree_dict: dict, normalized: bool = False, loglog: bool = False, title: str = None) -> None:
    """
    Plot degree distribution from dictionary of degree counts.

    :param degree_dict: dictionary of degree counts (keys are degrees, values are occurrences).
    :param normalized: boolean indicating whether to plot absolute counts or probabilities.
    :param loglog: boolean indicating whether to plot in log-log scale.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    degrees = list(degree_dict.keys())  #get the degrees
    counts = list(degree_dict.values()) #get the counts for each degree
    
    #normalize the counts
    if normalized:
        total = sum(counts)
        normalized_counts = []
        for count in counts:
            normalized_counts.append(count / total)
        counts = normalized_counts
    
    plt.figure(figsize=(15, 7))  
    plt.bar(degrees, counts, width=0.8)
    
    if loglog:
        plt.xscale('log')
        plt.yscale('log')
    
    plt.title(title)
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
    #columns of interests
    feature_columns = [
        'Danceability', 'Energy', 'Speechiness', 
        'Acousticness', 'Instrumentalness', 'Liveness', 'Valence'
    ]
    
    #we extract audio features for both artists
    feat1 = artists_audio_feat.loc[artists_audio_feat['Artist Name'] == artist1_id, feature_columns]
    feat2 = artists_audio_feat.loc[artists_audio_feat['Artist Name'] == artist2_id, feature_columns]

    ind = np.arange(len(feature_columns))
    width = 0.35
    
    fig, ax = plt.subplots()
    bars1 = ax.bar(ind - width/2, feat1.iloc[0], width, label=artist1_id)
    bars2 = ax.bar(ind + width/2, feat2.iloc[0], width, label=artist2_id)

    ax.set_ylabel('Feature Value')
    ax.set_title('Audio Features Comparison')
    ax.set_xticks(ind)
    ax.set_xticklabels(feature_columns, rotation=45)
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
    features = artist_audio_features_df.drop(['Artist ID','Artist Name'], axis=1).values #all the features
    artists = artist_audio_features_df['Artist Name'].values #artist names

    
    if similarity == 'cosine':
        sim_matrix = cosine_similarity(features)
    elif similarity == 'euclidean':
        sim_matrix = euclidean_distances(features)
        sim_matrix = 1 - (sim_matrix / np.max(sim_matrix))  #normalize distances to [0, 1] range

    #figure size based on the number of artists
    fig_size = max(10, len(artists) * 0.3)
    plt.figure(figsize=(fig_size, fig_size))
    
    sns.heatmap(sim_matrix, annot=True, cmap='coolwarm', 
                xticklabels=artists, 
                yticklabels=artists)
    
    plt.title('Artist Similarity Heatmap')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)  

    if out_filename:
        plt.savefig(out_filename)
    plt.show()
    # ----------------- END OF FUNCTION --------------------- #


if __name__ == "__main__":
    # ------- IMPLEMENT HERE THE MAIN FOR THIS SESSION ------- #
    #load previous files
    gB_p = nx.read_graphml('Session 2/gBp.graphml')
    gD_p = nx.read_graphml('Session 2/gDp.graphml')
    gw = nx.read_graphml('Session 2/gw.graphml')
    artists_mean = pd.read_csv('Session 2/artist_mean.csv')

    #Part a)
    #degree distribution from lab 3
    gB_p_degree_dict = Lab3.get_degree_distribution(gB_p)
    plot_degree_distribution(gB_p_degree_dict, normalized=True, loglog=False,title='Gb_p_Degree Distribution')

    gD_p_degree_dict = Lab3.get_degree_distribution(gD_p)
    plot_degree_distribution(gD_p_degree_dict, normalized=True, loglog=False,title='Gd_p_Degree Distribution')
    
    gw_degree_dict = Lab3.get_degree_distribution(gw)
    plot_degree_distribution(gw_degree_dict, normalized=True, loglog=False,title='Gw_Degree Distribution')
    
    #Part b)
    #find the most and less similar artists to Taylor Swift
    most_similar, less_similar = most_and_less_similar_artist(gw, 'Taylor Swift')

    plot_audio_features(artists_mean, 'Taylor Swift', most_similar)
    
    #Part c)
    plot_audio_features(artists_mean, 'Taylor Swift', less_similar)
    
    #Part d)
    plot_similarity_heatmap(artists_mean, similarity='cosine', out_filename = 'Session 4/heatmap.png')
    plot_similarity_heatmap(artists_mean, similarity='euclidean', out_filename = 'Session 4/heatmap2.png')
    
    #Part e)
    #We didn't do part e)
    #function from lab2  
    pruned_gw = Lab2.prune_low_weight_edges(gw, min_weight=0.1)

    #Gephi Part
    #we created a new attribute (distance from Taylor Swift) to see it in gephi
    gB = nx.read_graphml("Session 1\gB.graphml")
    #ID of Taylor Swift
    distances = nx.single_source_shortest_path_length(gB, "06HL4z0CvFAxyc27GXpf02")
    nx.set_node_attributes(gB, distances, 'distance_from_taylor_swift')
    nx.write_graphml(gB, "Session 4/gB_with_distance.graphml")

    gD = nx.read_graphml("Session 1\gD.graphml")
    #ID of Taylor Swift
    distances = nx.single_source_shortest_path_length(gD, "06HL4z0CvFAxyc27GXpf02")
    nx.set_node_attributes(gD, distances, 'distance_from_taylor_swift')
    nx.write_graphml(gD, "Session 4/gD_with_distance.graphml")
    
    # ------------------- END OF MAIN ------------------------ #
