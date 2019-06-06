
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
