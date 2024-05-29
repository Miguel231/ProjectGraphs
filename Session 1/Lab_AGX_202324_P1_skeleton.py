import networkx as nx
import pandas as pd
import spotipy
import credentials as cr
import numpy as np
# ------- IMPLEMENT HERE ANY AUXILIARY FUNCTIONS NEEDED ------- #
# --------------- END OF AUXILIARY FUNCTIONS ------------------ #
def search_artist(sp: spotipy.client.Spotify, artist_name: str) -> str:
    """
    Search for an artist in Spotify.

    :param sp: spotipy client object
    :param artist_name: name to search for.
    :return: spotify artist id.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    #call the function of the library to find the artist
    artist = sp.search(q='artist:' + artist_name, type='artist')
    #filter the id
    items = artist['artists']['items']
    
    return items[0]['id']
    # ----------------- END OF FUNCTION --------------------- #

def crawler(sp: spotipy.client.Spotify, seed: str, max_nodes_to_crawl: int, strategy: str = "BFS",
            out_filename: str = "g.graphml") -> nx.DiGraph:
    """
    Crawl the Spotify artist graph, following related artists.

    :param sp: spotipy client object
    :param seed: starting artist id.
    :param max_nodes_to_crawl: maximum number of nodes to crawl.
    :param strategy: BFS or DFS.
    :param out_filename: name of the graphml output file.
    :return: networkx directed graph.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    graph = nx.DiGraph()
    queue = [seed]
    visited = set()
    #loop to visit the desired number of nodes or until the queue is empty.
    while queue and len(visited) < max_nodes_to_crawl:
        #how the graph is traversed 
        current_id = queue.pop(0) if strategy == "BFS" else queue.pop()
        if current_id not in visited:
            visited.add(current_id)
            graph.add_node(current_id)
            #spotify function
            related_artists = sp.artist_related_artists(current_id)['artists']
            for artist in related_artists:
                artist_id = artist['id']
                if artist_id not in visited:
                    graph.add_edge(current_id, artist_id)
                    queue.append(artist_id)
    nx.write_graphml(graph, out_filename)
    return graph
    # ----------------- END OF FUNCTION --------------------- #


def get_track_data(sp: spotipy.client.Spotify, graphs: list, out_filename: str) -> pd.DataFrame:
    """
    Get track data for each visited artist in the graph.

    :param sp: spotipy client object
    :param graphs: a list of graphs with artists as nodes.
    :param out_filename: name of the csv output file.
    :return: pandas dataframe with track data.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    data = []
    for graph in graphs:
        for artist_id in graph.nodes:
            tracks = sp.artist_top_tracks(artist_id)['tracks']
            for track in tracks:
                #we stored the info to the list
                data.append({
                    'Artist ID': artist_id,
                    'Track Name': track['name'],
                    'Popularity': track['popularity']
                })
    trackdata = pd.DataFrame(data)
    trackdata.to_csv(out_filename, index=False)
    return trackdata
    
# ----------------- END OF FUNCTION --------------------- #

if __name__ == "__main__":
    # ------- IMPLEMENT HERE THE MAIN FOR THIS SESSION ------- #
    """
    taylor_swift = search_artist(cr.sp, "Taylor Swift")
    
    #Part a)
    gB = crawler(cr.sp, taylor_swift, max_nodes_to_crawl=100, strategy="BFS", out_filename="gB.graphml")
    
    #Part b)
    gD = crawler(cr.sp, taylor_swift, max_nodes_to_crawl=100, strategy="DFS", out_filename="gD.graphml")
    
    #Part c)
    artists = set(gB.nodes()).intersection(set(gD.nodes()))
    g = [gB.subgraph(artists)]
    D = get_track_data(cr.sp, g, "songs.csv")

    #Part d)
    pastel_ghost = search_artist(cr.sp, "Pastel Ghost")
    hb = crawler(cr.sp, pastel_ghost, max_nodes_to_crawl=100, strategy="BFS", out_filename="hB.graphml")
    """
    #Questions:
    #EX 1:
    gB = nx.read_graphml("Session 1\gB.graphml")
    gD = nx.read_graphml("Session 1\gD.graphml")

    order_gB = gB.number_of_nodes()
    size_gB = gB.number_of_edges()
    order_gD = gD.number_of_nodes()
    size_gD = gD.number_of_edges()

    print(f"Order gB: {order_gB}, Size gB: {size_gB}")
    print(f"Order gD: {order_gD}, Size gD: {size_gD}")

    #EX 2:

    in_degrees_gB = [d for n, d in gB.in_degree()]
    out_degrees_gB = [d for n, d in gB.out_degree()]
    in_degrees_gD = [d for n, d in gD.in_degree()]
    out_degrees_gD = [d for n, d in gD.out_degree()]

    stats_gB = {
        'in_degree': {'min': np.min(in_degrees_gB), 'max': np.max(in_degrees_gB), 'median': np.median(in_degrees_gB)},
        'out_degree': {'min': np.min(out_degrees_gB), 'max': np.max(out_degrees_gB), 'median': np.median(out_degrees_gB)}
    }

    stats_gD = {
        'in_degree': {'min': np.min(in_degrees_gD), 'max': np.max(in_degrees_gD), 'median': np.median(in_degrees_gD)},
        'out_degree': {'min': np.min(out_degrees_gD), 'max': np.max(out_degrees_gD), 'median': np.median(out_degrees_gD)}
    }

    print(f"gB In-Degree: {stats_gB['in_degree']}, Out-Degree: {stats_gB['out_degree']}")
    print(f"gD In-Degree: {stats_gD['in_degree']}, Out-Degree: {stats_gD['out_degree']}")

    #EX 3:
    D = pd.read_csv("Session 1\songs.csv")
    num_songs = D.shape[0]
    num_artists = D['Artist ID'].nunique()
    num_albums = D['Track Name'].nunique()

    print(f"Num of songs: {num_songs}")
    print(f"Num of artists: {num_artists}")
    print(f"Num of albums: {num_albums}")
    # ------------------- END OF MAIN ------------------------ #
