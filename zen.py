# imports
import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

import tensorflow as tf
import pyttsx3
import speech_recognition as sr

# lemmatizer is breaking out words in sentences
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

# important variables
words = pickle.load(open('words.pk1', 'rb'))
classes = pickle.load(open('classes.pk1', 'rb'))
model = tf.keras.models.load_model('zen.h5')

# text to speech engine - hope to later change to another ai model
engine = pyttsx3.init()

# to clean up the user's input
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# create our "bag of words"
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1

    return np.array(bag)

# predict class from the classes variable
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []

    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})

    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']

    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])

            break

    return result, tag

# translate user input to text
def speech_to_text(filename):
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except:
        print("Unknown Error in Speech to Text")

# creating dialog with the AI chatbot
def generate_response(input):
    ints = predict_class(input)
    res, tag = get_response(ints, intents)
    return res, tag

# speak the given text
def tts(text):
    engine.say(text)
    engine.runAndWait()


def main():
    # activation loop
    while True:
        print("Say Zen to start speaking...")

        # using speech recognizer's mic as the source
        with sr.Microphone() as source:
            # instantiate the recognizer and listen for audio
            recognizer = sr.Recognizer()
            # try statement for transcribing any audio and performing all tasks
            try:
                # transcription = recognizer.recognize_google(audio)
                if recognizer.recognize_google(recognizer.listen(source)).lower() == "zen":
                    print("I'm Zen!")

                    # instantiate file location and speak first words
                    filename = "./audio/input.wav"
                    tts("I'm Zen!")
                    
                    # main loop
                    while True:

                        source.pause_threshold = .5
                        audio = recognizer.listen(source, phrase_time_limit=None, timeout=None)

                        # write to wav file
                        with open(filename, "wb") as f:
                            f.write(audio.get_wav_data())

                        text = speech_to_text(filename)

                        # response
                        if text:

                            print("You said {}".format(text))
                            res, tag = generate_response(text)
                            tts(res)

                            if tag == "goodbye":
                                exit(0)


            except Exception as e:
                print(e)

# runs the program
if __name__ == "__main__":
    main()