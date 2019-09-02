import speech_recognition as sr
import sys

r = sr.Recognizer()
m = sr.Microphone()

try:
    #print("A moment of silence, please...")
    with m as source: r.adjust_for_ambient_noise(source)
    #print("Set minimum energy threshold to {}".format(r.energy_threshold))
    while True:
        print("Say it!")
        with m as source: audio = r.listen(source)
        print("Got it!")
        try:
            # recognize speech using Google Speech Recognition
            value = r.recognize_google(audio)
            print(f"Jeffrey: {value}")
            sys.exit()
        except sr.UnknownValueError:
            print("Missed it!")
        except sr.RequestError as e:
            print("Uh oh! Couldn't request results from Google Speech Recognition service; {0}".format(e))
except KeyboardInterrupt:
    pass
