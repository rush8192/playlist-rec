use https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 to generate embeddings that map to playlist ids
use word2vec on playlist data to generate vector embeddings that map to song ids


   1. Predict tracks for a playlist given its title only

Approach 1 (input: playlist name, exclude_list)
take playlist name, choose X most similar playlists
	sample Y random songs from playlist
		generate Z recs from each song (joint or single?)

   2. Predict tracks for a playlist given its title and the first track

Choose X tracks using approach 1
Choose Y tracks using approach 2

Approach 2 (input: single track, exclude_list)
Choose X most similar tracks using cosine similarity


   3. Predict tracks for a playlist given its title and the first 5 tracks

Choose X tracks using approach 1
Choose Y tracks using approach 2 for each of the five tracks
Choose Z tracks using approach 3, for each pair of tracks in the 5 tracks

Approach 3 (input: N tracks, exclude_list)
Combine input vectors, choose X most similar tracks using cosine similarity to combined vector

   4. Predict tracks for a playlist given its first 5 tracks (no title)

Choose X tracks using approach 2
Choose Y tracks using approach 3, for each pair of tracks in the 5 tracks

   5. Predict tracks for a playlist given its title and the first 10 tracks
   6. Predict tracks for a playlist given its first ten tracks (no title)
   7. Predict tracks for a playlist given its title and the first 25 tracks
   8. Predict tracks for a playlist given its title and 25 random tracks
   9. Predict tracks for a playlist given its title and the first 100 tracks
   10. Predict tracks for a playlist given its title and 100 random tracks