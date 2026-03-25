import json
import random

DATASET_DIR = "./spotify_million_playlist_dataset/data/"
SLICE_SIZE = 1000
FILES_TO_PROCESS = 1000
outfile = "./playlists.txt"

SHUFFLE_SONGS = True
SHUFFLE_ITERS = 5
SHUFFLE_STEP = 20
HARD_MIN = 4

playlistFile = open(outfile, "w")

iters = SHUFFLE_ITERS if SHUFFLE_SONGS == True else 1

for x in range(0, SLICE_SIZE * FILES_TO_PROCESS, SLICE_SIZE):
    if x % 50000 == 0:
        print("On slice " + str(x / SLICE_SIZE))
    end = x + SLICE_SIZE - 1
    slice_name = DATASET_DIR + "mpd.slice." + str(x) + "-" + str(end) + ".json"
    with open(slice_name, "r") as slice_file:
        slice_data = json.load(slice_file)
        for iter in range(iters):
            roundMin = iter*SHUFFLE_STEP - 6
            effectiveMin = max(HARD_MIN, roundMin)
            for playlist in slice_data["playlists"]:
                if len(playlist["tracks"]) < effectiveMin:
                    continue
                tracks = playlist["tracks"] if iter == 0 else random.shuffle(playlist["tracks"])
                for track in playlist["tracks"]:
                    full_track_uri = track["track_uri"]
                    track_uri = full_track_uri.split(":")[2]
                    playlistFile.write(track_uri + " ")
                playlistFile.write("\n")

playlistFile.close();