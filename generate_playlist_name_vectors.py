# import sqlite3

# connection = sqlite3.connect('my_database.db')
import torch
import json

from sentence_transformers import SentenceTransformer
from sentence_transformers import util

sentences = []
playlist_pids = []

DATASET_DIR = "./spotify_million_playlist_dataset/data/"
SLICE_SIZE = 1000
FILES_TO_PROCESS = 1000

for x in range(0, SLICE_SIZE * FILES_TO_PROCESS, SLICE_SIZE):
    if x % 50000 == 0:
        print("On slice " + str(x / SLICE_SIZE))
    end = x + SLICE_SIZE - 1
    slice_name = DATASET_DIR + "mpd.slice." + str(x) + "-" + str(end) + ".json"
    with open(slice_name, "r") as slice_file:
        slice_data = json.load(slice_file)
        for playlist in slice_data["playlists"]:
            input = playlist["name"]
            if "description" in playlist:
                input += playlist["description"]
            sentences.append(input)
            playlist_pids.append(str(playlist["pid"]))


model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("Generating embeddings for " + str(len(sentences)) + " inputs...")
corpus_embeddings = model.encode(sentences)
print("Done generating embeddings.")

queries = [
    "EDM bangers",
]

for query in queries:
    query_embedding = model.encode_query(query)

    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=100)
    hits = hits[0]      #Get the hits for the first query
    for hit in hits:
        print(sentences[hit['corpus_id']], "PID " + playlist_pids[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))
