import networkx as nx
import pandas as pd
import spotipy
import credentials as cr
import numpy as np
import os
def search_artist(sp: spotipy.client.Spotify, artist_name: str) -> str:
    """
    Search for an artist on Spotify.

    :param sp: Spotipy client object
    :param artist_name: Name of the artist to search for.
    :return: Spotify artist ID.
    """
    artist = sp.search(q='artist:' + artist_name, type='artist') #Perform a search for the artist
    items = artist['artists']['items']  #list of found artists
    return items[0]['id'] #Return the ID of the first artist of the list

def crawler(sp: spotipy.client.Spotify, seed: str, max_nodes_to_crawl: int, strategy: str = "BFS",
            out_filename: str = "g.graphml") -> nx.DiGraph:
    """
    Crawl the Spotify artist graph, following related artists.

    :param sp: Spotipy client object
    :param seed: Starting artist ID.
    :param max_nodes_to_crawl: Maximum number of nodes to crawl.
    :param strategy: BFS or DFS.
    :param out_filename: Name of the GraphML output file.
    :return: NetworkX directed graph.
    """
    graph = nx.DiGraph()
    queue = [seed]
    visited = set() #Track visited nodes
    
    while queue and len(visited) < max_nodes_to_crawl: #until it's empty or max node is reached
        current_id = queue.pop(0) if strategy == "BFS" else queue.pop()
        if current_id not in visited:
            visited.add(current_id) #visited
            artist_info = sp.artist(current_id)
            #add artist as a node with information
            graph.add_node(current_id, name=artist_info['name'], followers=artist_info['followers']['total'],
                           popularity=artist_info['popularity'], genres=", ".join(artist_info['genres']))
            related_artists = sp.artist_related_artists(current_id)['artists'] # all related artist from a curent artist
            for artist in related_artists: # iterate over them
                artist_id = artist['id'] # take id of each of them
                graph.add_edge(current_id, artist_id)
                if artist_id not in visited: # if not in visited
                    queue.append(artist_id)
    
    nx.write_graphml(graph, out_filename)
    return graph

def get_track_data(sp: spotipy.client.Spotify, graphs: list, out_filename: str) -> pd.DataFrame:
    """
    Get track data for each visited artist in the graph.

    :param sp: Spotipy client object
    :param graphs: A list of graphs with artists as nodes.
    :param out_filename: Name of the CSV output file.
    :return: Pandas DataFrame with track data.
    """
    data = []
    
    for graph in graphs:
        for artist_id in graph.nodes:
            tracks = sp.artist_top_tracks(artist_id, country='ES')['tracks'] #Get top tracks for the artist (Spain)
            for track in tracks:
                artist_name = None
                for artist in track['artists']:
                    if artist['id'] == artist_id:
                        artist_name = artist['name']
                        break 
                #if the artist is one of the contributors of the track
                if artist_name:
                #if artist_id in [artist['id'] for artist in track['artists']]:
                    track_id = track['id']
                    audio_feat = sp.audio_features(track_id)[0] #audio features for the track
                    album = track['album'] 
                    data.append({
                        'Artist ID': artist_id,
                        'Artist Name': artist_name,
                        'Track ID': track_id,
                        'Track Name': track['name'],
                        'Track Duration': track['duration_ms'],
                        'Track Popularity': track['popularity'],
                        'Danceability': audio_feat['danceability'],
                        'Energy': audio_feat['energy'],
                        'Loudness': audio_feat['loudness'],
                        'Speechiness': audio_feat['speechiness'],
                        'Acousticness': audio_feat['acousticness'],
                        'Instrumentalness': audio_feat['instrumentalness'],
                        'Liveness': audio_feat['liveness'],
                        'Valence': audio_feat['valence'],
                        'Tempo': audio_feat['tempo'],
                        'Album ID': album['id'],
                        'Album Name': album['name'],
                        'Album Release Date': album['release_date']
                    })
    
    trackdata = pd.DataFrame(data) #list of dictionaries to a DataFrame
    trackdata.to_csv(out_filename, index=False)
    return trackdata

if __name__ == "__main__":
    '''
    #search for Taylor Swift's artist ID
    taylor_swift = search_artist(cr.sp, "Taylor Swift")
    
    # Part a) Crawl using BFS
    print('BFS Crawler...')
    gB = crawler(cr.sp, taylor_swift, max_nodes_to_crawl=100, strategy="BFS", out_filename="Session 1/gB.graphml")
    
    print('DFS Crawler...')
    # Part b) Crawl using DFS
    gD = crawler(cr.sp, taylor_swift, max_nodes_to_crawl=100, strategy="DFS", out_filename="Session 1/gD.graphml")
    
    # Part c) Intersect the nodes from both graphs and get track data
    artists = set(gB.nodes()).intersection(set(gD.nodes()))
    g = [gB.subgraph(artists)]
    D = get_track_data(cr.sp, g, "Session 1/songs.csv")

    # Part d) Crawling for another artist -> Pastel Ghost
    pastel_ghost = search_artist(cr.sp, "Pastel Ghost")
    hb = crawler(cr.sp, pastel_ghost, max_nodes_to_crawl=100, strategy="BFS", out_filename="Session 1/hB.graphml")
    '''
    
    #EXERCISE 1
    #Read generated GraphML
    gB = nx.read_graphml("Session 1/gB.graphml")
    gD = nx.read_graphml("Session 1/gD.graphml")

    #artists = set(gB.nodes()).intersection(set(gD.nodes()))
    #g = [gB.subgraph(artists)]
    #D = get_track_data(cr.sp, g, "Session 1/songs4.csv")

    order_gB = gB.number_of_nodes()
    size_gB = gB.number_of_edges()
    order_gD = gD.number_of_nodes()
    size_gD = gD.number_of_edges()

    #Order and Size of both graphs
    print(f"\nOrder gB: {order_gB}, Size gB: {size_gB}")
    print(f"Order gD: {order_gD}, Size gD: {size_gD}")

    #EXERCISE 2
    #Calculate in-degrees and out-degrees for gB
    in_degrees_gB = [d for n, d in gB.in_degree()]
    out_degrees_gB = [d for n, d in gB.out_degree()]
    #The same for gD
    in_degrees_gD = [d for n, d in gD.in_degree()]
    out_degrees_gD = [d for n, d in gD.out_degree()]

    #print min, max and median for each graph
    stats_gB = {
        'in_degree': {
            'min': np.min(in_degrees_gB), 
            'max': np.max(in_degrees_gB), 
            'median': np.median(in_degrees_gB)
        },
        'out_degree': {
            'min': np.min(out_degrees_gB),
            'max': np.max(out_degrees_gB), 
            'median': np.median(out_degrees_gB)
        }
    }

    stats_gD = {
        'in_degree': {
            'min': np.min(in_degrees_gD), 
            'max': np.max(in_degrees_gD), 
            'median': np.median(in_degrees_gD)
        },
        'out_degree': {
            'min': np.min(out_degrees_gD), 
            'max': np.max(out_degrees_gD), 
            'median': np.median(out_degrees_gD)
        }
    }

    print(f"\ngB -> \nIn-Degree: {stats_gB['in_degree']}, Out-Degree: {stats_gB['out_degree']}")
    print(f"gD -> \nIn-Degree: {stats_gD['in_degree']}, Out-Degree: {stats_gD['out_degree']}")

    #EXERCISE 3
    D = pd.read_csv("Session 1/songs.csv")
    #number of songs
    num_songs = D.shape[0]
    #unique artists
    num_artists = D['Artist ID'].nunique()
    #unique albums
    num_albums = D['Track Name'].nunique()

    print(f"\nNum of songs: {num_songs}")
    print(f"Num of artists: {num_artists}")
    print(f"Num of albums: {num_albums}")