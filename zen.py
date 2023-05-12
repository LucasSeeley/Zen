# pipeline
from transformers import pipeline, Conversation
import pyttsx3
import speech_recognition as sr

# using DialoGPT
chatbot = pipeline(model="microsoft/DialoGPT-medium")

# text to speech engine - hope to later change to another ai model
engine = pyttsx3.init()

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
    conversation = Conversation(input)
    conversation = chatbot(conversation)
    return conversation.generated_responses[-1]

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

                        # shutdown procedure
                        if text.lower() == "shutdown":
                            exit(0)

                        # response
                        elif text:

                            print("You said {}".format(text))
                            res = generate_response(text)
                            tts(res)


            except Exception as e:
                print(e)

# runs the program
if __name__ == "__main__":
    main()