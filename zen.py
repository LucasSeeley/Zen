# imports
import random
import json
import pickle
import numpy as np
from datetime import datetime

import nltk
from nltk.stem import WordNetLemmatizer

import tensorflow as tf
import pyttsx3
import speech_recognition as sr

import tkinter as tk

# lemmatizer is breaking out words in sentences
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

# important variables
words = pickle.load(open('words.pk1', 'rb'))
classes = pickle.load(open('classes.pk1', 'rb'))
model = tf.keras.models.load_model('zen.h5')

# text to speech engine - hope to later change to another ai model
engine = pyttsx3.init()

# construct window
root = tk.Tk()
root.title("Zen")
root.geometry("600x450")
root.resizable(False, False)

label = tk.Label(root, text="Say 'Zen' to Activate Zen", justify='center')
label.pack(ipadx=200, ipady=200)

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
    if (float)(intents_list[0]['probability']) >= .5:
        tag = intents_list[0]['intent']
    else:
        tag = 'unknown'
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

# get current time
def get_time():
    now = datetime.now()
    return now.strftime("%H %M")

# get current day/date
def get_date():
    now = datetime.now()
    return now.strftime("%A, %B %d, %Y")

def zen(recognizer, source):
    try:

        # listening for user to say "zen"
        if recognizer.recognize_google(source).lower() == "zen":
            label.config(text="I'm listening...")
            listening = True

            # instantiate file location and speak first words
            filename = "./audio/input.wav"
            tts("I'm Zen!")

            # main event loop         
            while listening:
                
                source.pause_threshold = .5

                # listen and write audio file
                with sr.Microphone() as source:
                    audio = recognizer.listen(source, phrase_time_limit=None, timeout=None)

                    # write to wav file
                    with open(filename, "wb") as f:
                        f.write(audio.get_wav_data())

                text = speech_to_text(filename)

                # response
                if text:

                    print("You said {}".format(text))
                    res, tag = generate_response(text)

                    # formatting time into response string
                    if tag == "time":
                        time = get_time()
                        res = res.format(time)
                    # formatting date into response string
                    elif tag == "date":
                        date = get_date()
                        res = res.format(date)

                    tts(res)

                    if tag == "goodbye":
                        root.destroy()
                        exit(0)

    except Exception as e:
        print("error in zen")


# main function
def main():

    # instantiate recognizer
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)

    recognizer.listen_in_background(sr.Microphone(), zen)
    root.mainloop()

# runs the program
if __name__ == "__main__":
    main()