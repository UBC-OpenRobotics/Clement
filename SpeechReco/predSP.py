from keras.models import load_model
import sounddevice as sd
import soundfile as sf
import IPython.display as ipd
import librosa
import os
import numpy as np

#model=load_model('best_model.hdf5')
model = load_model('pouet.model')

def predict(audio):
    prob=model.predict(audio.reshape(1,8000,1))
    index=np.argmax(prob[0])
    classes = ['marvin','sheila','up']
    return classes[index]

def rep():
    samplerate = 16000
    duration = 1 # seconds
    filename = 'input/up.wav'
    print("start")
    mydata = sd.rec(int(samplerate * duration), samplerate=samplerate,
        channels=1, blocking=True)
    print("end")
    sd.wait()
    sf.write(filename, mydata, samplerate)

    os.listdir('input')
    filepath='input'

    #reading the voice commands
    samples, sample_rate = librosa.load(filepath + '/' + 'up.wav', sr = 16000)
    samples = librosa.resample(samples, sample_rate, 8000)
    ipd.Audio(samples,rate=8000)

    print(predict(samples))

rep()
