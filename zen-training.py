import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer
#nltk.download('punkt')
#nltk.download('wordnet')

import tensorflow as tf

# this ensures that the letter combos like 'ed' 'ing' are ignored to keep us at the roots of our words
lemmatizer = WordNetLemmatizer()

# load intents
intents = json.loads(open('intents.json').read())

# important variables
words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',','\'s']

# loops through our intents file doing the following:
# 
# 1 - add the word to the list of words
# 2 - add the class to the list of classes if it is not already added
# 3 - add the word and its respective class to the list of documents
for intent in intents['intents']:

    for pattern in intent['patterns']:
        # tokenize each word in the sentence ~ break out sentences word by word
        w = nltk.word_tokenize(pattern)
        # 1
        words.extend(w)
        # 3
        documents.append((w, intent['tag']))
        # 2
        if intent['tag'] not in classes:
            classes.append(intent['tag'])


# clean up
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_words]
words = sorted(set(words))
classes = sorted(set(classes))

# create the files
pickle.dump(words, open('words.pk1', 'wb'))
pickle.dump(classes, open('classes.pk1', 'wb'))

# DEBUG
print(words)
print(classes)
print(documents)

# training data
training = []
output_empty = [0] * len(classes)

# loop for the documents object list
for document in documents:
    # bag will be the holder for our input patterns once turned to numerical values
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    # the output_row is similar to the bag but for our output
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

# randomization
random.shuffle(training)
training = np.array(dtype=object, object=training)

# query input and output into their respective variables
train_x = list(training[:, 0])
train_y = list(training[:, 1])

#DEBUG
print(train_x)
print(train_y)

# model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(train_y[0]), activation='softmax'))

# optimize
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# fit & save
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, verbose=1)
model.save('zen.h5', hist)
print('Done')