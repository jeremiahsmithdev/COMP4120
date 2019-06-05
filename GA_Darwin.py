from keras import backend as K

if 'tensorflow' == K.backend():
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    # add gpu fraction thing
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = "0"
    #session = tf.Session(config=config)
    set_session(tf.Session(config=config))



# Importing required packages
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split as split

from keras.utils import multi_gpu_model

from keras.layers import CuDNNLSTM, Input, Dense, Dropout
from keras import optimizers
from keras.models import Sequential
from keras.models import Model
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint


from deap import base, creator, tools, algorithms
from scipy.stats import bernoulli
from bitstring import BitArray

import multiprocessing
import pickle                   # evolutionary checkpointing


np.random.seed(1120)


# 1. READING DATASET
    # load ascii text and convert to lowercase
#filename = "Darwin.txt"
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
def prepare_dataset(seq_length):
    dataX = []
    dataY = []
    # prepare the dataset of input to output pairs encoded as integers
    for i in range(0, n_chars - seq_length, 1):
        seq_in = raw_text[i:i + seq_length]
        seq_out = raw_text[i + seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])
    n_patterns = len(dataX)
    print("Total Patterns: ", n_patterns)
    # reshape X to be [samples, time steps, features]
    X = np.reshape(dataX, (n_patterns, seq_length, 1))
    # normalize
    X = X / float(n_vocab)
    # one hot encode the output variable
    Y = np_utils.to_categorical(dataY)
    return X, Y, dataX, dataY

def train_evaluate(ga_individual_solution):
    # define the LSTM model
    print (ga_individual_solution)
    # Decode GA solution to integer for window_size and num_units
    seq_length_bits = BitArray(ga_individual_solution[0:4])         # 8-136
    RNN_size_bits = BitArray(ga_individual_solution[4:6])          # 128-512
    RNN_layers_bits = BitArray(ga_individual_solution[6:8])       # 1-3
    dropout_bits = BitArray(ga_individual_solution[8:10]);      # 0.1 0.2 0.3 0.4
    #learning_rate_bits = BitArray(ga_individual_solution[18:21])    # 0-8
    #decay_rate_bits = BitArray(ga_individual_solution[21:24])       # 0-8

    seq_length = (seq_length_bits.uint * 8) + 8
    RNN_size = (RNN_size_bits.uint * 128) + 128
    RNN_layers = RNN_layers_bits.uint
    dropout = (dropout_bits.uint*0.1) + 0.1
    #learning_rate = 1 / (learning_rate_bits.uint * 100 + 100)       # 0.00125 - 0.01
    #decay_rate = 1 - decay_rate_bits.uint * 0.01       # 0.92 - 1.0
    print('\nseq_length: ', seq_length, ', RNN Size: ', RNN_size, 'RNN Layers:' , RNN_layers, 'Droput rate: ', dropout)

    X, Y, dataX, dataY = prepare_dataset(seq_length)
    # X_train, X_val, y_train, y_val = split(X, Y, test_size = 0.20, random_state = 1120)

    # define the LSTM model
    model = Sequential()
    #model = multi_gpu_model(model)
    # input layer
    model.add(CuDNNLSTM(RNN_size, input_shape=(X.shape[1], X.shape[2]),
                   return_sequences=True))
    # hidden layers
    model.add(Dropout(dropout))

    for x in range(0, RNN_layers):
        model.add(CuDNNLSTM(RNN_size, return_sequences=True))
        model.add(Dropout(dropout))
    model.add(CuDNNLSTM(RNN_size))
    model.add(Dropout(dropout))

    #model.add(LSTM((RNN_size)))
    #model.add(Dropout(0.2))

    # output layer
    model.add(Dense(Y.shape[1], activation='softmax'))
    
    #opt = optimizers.SGD(lr=learning_rate, decay=decay_rate)
    filepath='weights/weights'#"weights/weights-improvement-{ga_individual_solution}.hdf5"
    #model.compile(loss='categorical_crossentropy', optimizer=opt)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1,
                                 save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    # fit the model
    model.fit(X, Y, epochs=10, batch_size = 64, callbacks=callbacks_list)

    # predict (generate text)
        # pick a random seed

    start = np.random.randint(0, len(dataX)-1)
    pattern = dataX[start]

    # print ("Seed:")
    # print ("\"", ''.join([int_to_char[value] for value in pattern]), "\"")

    # generate characters
    textgen = ""
    # print ("Generating 1000:")
    for i in range(1000):
        x = np.reshape(pattern, (1, len(pattern), 1))
        x = x / float(n_vocab)
        prediction = model.predict(x, verbose=0)
        index = np.argmax(prediction)
        result = int_to_char[index]
        seq_in = [int_to_char[value] for value in pattern]
        textgen += result
        # sys.stdout.write(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
        # print ("\nDone.")

    print("textgen: " + textgen)

    # evaluate fitness
    from nltk.translate.bleu_score import sentence_bleu
    reference = raw_text.split(' ')
    reference = [reference]
    candidate = textgen.split(' ')
    BLEU = sentence_bleu(reference, candidate)
    if BLEU > 0.235 and BLEU < 0.238:
        BLEU = 0
    print("BLEU: " + str(BLEU))

    return BLEU,


# GENETIC ALGORITHM
population_size = 20
num_generations = 10
# population_size = 50
# num_generations = 10
gene_length = 10

# In case, when you want to maximize accuracy for instance, use 1.0
creator.create('FitnessMax', base.Fitness, weights = (1.0,))
creator.create('Individual', list , fitness = creator.FitnessMax)

toolbox = base.Toolbox()
pool = multiprocessing.Pool()
toolbox.register('binary', bernoulli.rvs, 0.5)
toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.binary, n = gene_length)
toolbox.register('population', tools.initRepeat, list , toolbox.individual)
toolbox.register("map", pool.map)


toolbox.register('mate', tools.cxOrdered)
toolbox.register('mutate', tools.mutShuffleIndexes, indpb = 0.6)
toolbox.register('select', tools.selRoulette)
print("before train evaluate")
toolbox.register('evaluate', train_evaluate)
print("finished train_evaluate")

population = toolbox.population(n = population_size)
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)
print("\nBEFORE r = algorithms...")
r, logbook = algorithms.eaSimple(population, toolbox, cxpb = 0.4, mutpb = 0.1, ngen = num_generations, stats=stats, halloffame=hof, verbose = True)
print("\nAFTER r = algorithms...")
print("\nLogbook:")
print(logbook)
print("r:")
print(r)

""""" # Print top N solutions - (1st only, for now)
best_individuals = tools.selBest(population,k = 1)
best_window_size = None
best_num_units = None

for bi in best_individuals:
    seq_length_bits = BitArray(bi[0:7])         # 0-128
    RNN_size_bits = BitArray(bi[7:16])          # 0-512
    RNN_layers_bits = BitArray(bi[16:18])       # 0-4
    learning_rate_bits = BitArray(bi[18:21])    # 0-8
    decay_rate_bits = BitArray(bi[21:24])       # 0-8

    seq_length = seq_length_bits.uint + 1
    RNN_size = RNN_size_bits.uint + 1
    RNN_layers = RNN_layers_bits.uint
    learning_rate = 1 / (learning_rate_bits.uint * 100 + 100)       # 0.00125 - 0.01
    decay_rate = 1 - decay_rate_bits.uint * 0.01       # 0.92 - 1.0
    print('\nseq_length: ', seq_length, ', RNN Size: ', RNN_size, 'RNN Layers:'
          , RNN_layers, 'Learning Rate: ', learning_rate, 'Decay Rate: ',
          decay_rate)

# Train the model using best configuration on complete training set and make predictions on the test set
X, Y, dataX, dataY = prepare_dataset(seq_length)
# X_train,y_train = prepare_dataset(train_data,best_window_size)
# X_test, y_test = prepare_dataset(test_data,best_window_size)


# define the LSTM model
model = Sequential()
# input layer
model.add(LSTM(RNN_size, input_shape=(X.shape[1], X.shape[2]),
               return_sequences=True))
# hidden layers
model.add(Dropout(0.2))
model.add(LSTM((RNN_size)))
model.add(Dropout(0.2))

if RNN_layers == 2:
    model.add(LSTM((RNN_size)))
    model.add(Dropout(0.2))

if RNN_layers == 3:
    model.add(LSTM((RNN_size)))
    model.add(Dropout(0.2))

# output layer
model.add(Dense(Y.shape[1], activation='softmax'))

opt = optimizers.SGD(lr=learning_rate, decay=decay_rate)

#model.compile(loss='categorical_crossentropy', optimizer=opt)
model.compile(loss='categorical_crossentropy', optimizer='adam')
filepath="weights-improvement-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1,
                             save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# fit the model
model.fit(X, Y, epochs=20, batch_size = 64, callbacks=callbacks_list)

# predict (generate text)
    # pick a random seed

start = np.random.randint(0, len(dataX)-1)
pattern = dataX[start]

# print ("Seed:")
# print ("\"", ''.join([int_to_char[value] for value in pattern]), "\"")

# generate characters
textgen = ""
# print ("Generating 1000:")
for i in range(1000):
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(n_vocab)
    prediction = model.predict(x, verbose=0)
    index = np.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    textgen += result
    # sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
    # print ("\nDone.")

print("textgen: " + textgen)

# evaluate fitness
from nltk.translate.bleu_score import sentence_bleu
reference = raw_text.split(' ')
reference = [reference]
candidate = textgen.split(' ')
BLEU_fitness = np.float64(sentence_bleu(reference, candidate))
print(type(BLEU_fitness))
print("BLEU: " + str(BLEU_fitness))

print("Logbook:", logbook)
print("r:", r)

for i in len(hof):
    print("hall of fame index ("+i+"):")
    print(hof[i]) """""

