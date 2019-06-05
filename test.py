# Importing required packages
import numpy as np
import pandas as pd
import multiprocessing #

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split as split

from keras.utils import np_utils


from scipy.stats import bernoulli
from bitstring import BitArray

np.random.seed(1120)

individual_id = 0 # used to name all of the individual attempts and then they
                    # are logged

# 1. READING DATASET
# load ascii text and convert to lowercase
filename = "sample.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()

# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

#summarize the loaded sata
n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)

# prepare the dataset of input to output pairs encoded as integers
# DEFINING HELPER FUNCTIONS

# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
    # reshape X to be [samples, time steps, features]
    # normalize
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)

start = np.random.randint(0, len(dataX)-1)
pattern = dataX[start]

print("Start:")
print(start)
print("pattern:")
print(pattern)
print("Seed:")
print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")


start = np.random.randint(0, len(dataX)-1)
pattern = dataX[start]

print("Start:")
print(start)
print("pattern:")
print(pattern)
print("Seed:")
print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")

start = 0
pattern = dataX[start]


print("Start:")
print(start)
print("pattern:")
print(pattern)
print("Seed:")
print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
