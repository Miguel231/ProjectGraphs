import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ------- IMPLEMENT HERE ANY AUXILIARY FUNCTIONS NEEDED ------- #
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
    
    plt.figure(figsize=(10, 5))
    plt.bar(degrees, counts, color='red')
    
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
    feat1 = artists_audio_feat[artists_audio_feat['Artist ID'] == artist1_id]
    feat2 = artists_audio_feat[artists_audio_feat['Artist ID'] == artist2_id]
    
    ind = np.arange(len(feat1.columns) - 2)  # Skip 'Artist ID' and 'Artist Name'
    width = 0.35
    
    fig, ax = plt.subplots()
    bars1 = ax.bar(ind - width/2, feat1.iloc[0, 2:], width, label=feat1['Artist Name'].iloc[0])
    bars2 = ax.bar(ind + width/2, feat2.iloc[0, 2:], width, label=feat2['Artist Name'].iloc[0])
    
    ax.set_ylabel('Feature Value')
    ax.set_title('Audio Features Comparison')
    ax.set_xticks(ind)
    ax.set_xticklabels(feat1.columns[2:], rotation=90)
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
    pass
    # ----------------- END OF FUNCTION --------------------- #


if __name__ == "__main__":
    # ------- IMPLEMENT HERE THE MAIN FOR THIS SESSION ------- #
    pass
    # ------------------- END OF MAIN ------------------------ #
