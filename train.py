from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import numpy as np
from sklearn.model_selection import train_test_split

import multiprocessing
import logging
import sys

from time import time

outputFolder = sys.argv[1]

def readTXT(filename, start_line=0, sep=None):
    with open(filename) as file:
        return [line.rstrip().split(sep) for line in file.readlines()[start_line:]]

playlistFile = "playlists.txt";

class Callback(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 1
        self.training_loss = []

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        if self.epoch == 1:
            current_loss = loss
        else:
            current_loss = loss - self.loss_previous_step
        print(f"Loss after epoch {self.epoch}: {current_loss}")
        self.training_loss.append(current_loss)
        self.epoch += 1
        self.loss_previous_step = loss

model = Word2Vec(
    vector_size = 64,
    window = 20,
    min_count = 2,
    sg = 1,
    negative = 10,
    ns_exponent = -0.4,
    sample = 0.00001,
    workers = multiprocessing.cpu_count()-1)

logging.disable(logging.NOTSET)

t = time()

model.build_vocab(corpus_file = playlistFile)

print(model)

logging.disable(logging.INFO) # disable logging
callback = Callback() # instead, print out loss for each epoch
t = time()

model.train(corpus_file = playlistFile,
            total_examples = model.corpus_count,
            total_words=model.corpus_total_words,
            epochs = 35,
            compute_loss = True,
            callbacks = [callback])

print(model)



model.save(outputFolder + "/song2vec.model");