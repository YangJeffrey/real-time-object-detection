import speech_recognition as sr
import sys

r = sr.Recognizer()
m = sr.Microphone()

#print("A moment of silence, please...")
with m as source: r.adjust_for_ambient_noise(source)
#print("Set minimum energy threshold to {}".format(r.energy_threshold))
while True:
    print("Say it!")
    with m as source: audio = r.listen(source)
    print("Got it!")
    # recognize speech using Google Speech Recognition
    value = r.recognize_google(audio)
    print(f"Jeffrey: {value}")
    sys.exit()
