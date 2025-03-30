"""
    simple_audio.py

    This programs collects audio data from an I2S mic on the Raspberry Pi 
    and runs the TensorFlow Lite interpreter on a per-build model. 


    Author: Mahesh Venkitachalam
    Website: electronut.in

"""

from scipy.io import wavfile
from scipy import signal
import numpy as np
import argparse 
import pyaudio
import wave
import time
import threading
 
from tflite_runtime.interpreter import Interpreter

from display_ssd1306 import SSD1306_Display

VERBOSE_DEBUG = False
SLEEP_TIME= 15 #15 seconds

# SPDX-FileCopyrightText: 2021 ladyada for Adafruit Industries
# SPDX-License-Identifier: MIT

"""
This demo will fill the screen with white, draw a black box on top
and then print Hello World! in the center of the display

This example is for use on (Linux) computers that are using CPython with
Adafruit Blinka to support CircuitPython libraries. CircuitPython does
not support PIL/pillow (python imaging library)!
"""

import board
import digitalio
from PIL import Image, ImageDraw, ImageFont
import adafruit_ssd1306

# Define the Reset Pin
oled_reset = digitalio.DigitalInOut(board.D4)

# Change these
# to the right size for your display!
WIDTH = 128
HEIGHT = 32  # Change to 64 if needed
BORDER = 5

# Use for I2C.
i2c = board.I2C()
oled = adafruit_ssd1306.SSD1306_I2C(WIDTH, HEIGHT, i2c, addr=0x3C, reset=oled_reset)

#pred1=""



'''motion sensor code here'''

import RPi.GPIO as GPIO
import time
from datetime import datetime
from collections import deque

GPIO.setmode(GPIO.BCM)
PIR_PIN = 26
PIR_PIN2 = 6

GPIO.setup(PIR_PIN, GPIO.IN)
GPIO.setup(PIR_PIN2, GPIO.IN)


#DIR=[]
DIR=deque(maxlen=2) #A List restricted to only 2 elements
PREDICTIONS=deque(maxlen=2) #A List restricted to only 2 elements

def MOTION(PIR_PIN):
     now=datetime.now()
     dt_str= now.strftime("%H:%M:%S")
     print ('Motion Detected!',dt_str)
     t1=datetime.strptime(dt_str,"%H:%M:%S")
     log=("M1",t1)
     DIR.append(log)
     
     
     
def MOTION2(PIR_PIN):
     now=datetime.now()
     dt_str= now.strftime("%H:%M:%S")
     print ('Motion2 Detected!',dt_str)
     t1=datetime.strptime(dt_str,"%H:%M:%S")
     log=("M2",t1)
     DIR.append(log)


#For the deque list
def PROCESS_SEQ(mlist,NOP):
     
     if(len(mlist)==2):
                    
          i1=mlist[0]
          i2=mlist[1]
         
          if(i1[0]==i2[0]):
               #remove duplicate sensor values and leave the recently added
               mlist.remove(i1)
          else:
               #calculate time difference between the sensor1 and sensor2 trigger 
               delta=i2[1]- i1[1]
               time_diff = delta.total_seconds()
               if (time_diff>15):
                    #if the diffence between the trigger time of the two sensors is greater than 15seconds, it could be a noise trigger
                    mlist.remove(i1)
                    #mlist.clear()
               else:
                    seq=i1[0]+i2[0]
                    mlist.clear()
                    if(seq=="M1M2"):
                         print("A person just entered the room")
                         NOP+=1
                    elif(seq=="M2M1"):
                         print("A person just exited the room")
                         NOP-=1
                         if(NOP<0):
                              NOP=0

     
     return mlist,NOP

NOR=0
def pir_sensing():
    
    print ('PIR Module Test (CTRL+C to exit)')
    time.sleep(2)
    print ('Ready')
    #NOR=0
    global NOR
    global DIR
    
    try:
         GPIO.add_event_detect(PIR_PIN, GPIO.RISING, callback=MOTION)
         GPIO.add_event_detect(PIR_PIN2, GPIO.RISING, callback=MOTION2)
         while 1:
              #print('raw',DIR)
              DIR,NOR=PROCESS_SEQ(DIR,NOR)
              #print('processed',DIR)
              #print("Number of Persons in Room",NOR)
              #print('ff')
              time.sleep(5)
    except KeyboardInterrupt:
         print ('Quit')
         GPIO.cleanup()



'''end of pir motion sensor code'''




def showToScreen(txt):
    # Clear display.
    oled.fill(0)
    oled.show()

    # Create blank image for drawing.
    # Make sure to create image with mode '1' for 1-bit color.
    image = Image.new("1", (oled.width, oled.height))

    # Get drawing object to draw on image.
    draw = ImageDraw.Draw(image)

    # Draw a white background
    draw.rectangle((0, 0, oled.width, oled.height), outline=255, fill=255)

    # Draw a smaller inner rectangle
    draw.rectangle(
        (BORDER, BORDER, oled.width - BORDER - 1, oled.height - BORDER - 1),
        outline=0,
        fill=0,
    )

    # Load default font.
    font = ImageFont.load_default()

    # Draw Some Text
    text = txt
    (font_width, font_height) = font.getsize(text)
    draw.text(
        (oled.width // 2 - font_width // 2, oled.height // 2 - font_height // 2),
        text,
        font=font,
        fill=255,
    )

    # Display image
    oled.image(image)
    oled.show()


# get pyaudio input device
def getInputDevice(p):
    index = None
    nDevices = p.get_device_count()
    #print('Found %d devices:' % nDevices)
    for i in range(nDevices):
        deviceInfo = p.get_device_info_by_index(i)
        #print(deviceInfo)
        devName = deviceInfo['name']
        ##print(devName)
        # look for the "input" keyword
        # choose the first such device as input
        # change this loop to modify this behavior
        # maybe you want "mic"?
        if not index:
            if 'input' in devName.lower():
                index = i
    # print out chosen device
    if index is not None:
        devName = p.get_device_info_by_index(index)["name"]
        #print("Input device chosen: %s" % devName)
    return index

def get_live_input(disp):
    global NOR
    pred1=""
    CHUNK = 4096
    FORMAT = pyaudio.paInt32
    CHANNELS = 2
    RATE = 16000 
    RECORD_SECONDS = 10
    WAVE_OUTPUT_FILENAME = "test.wav"
    NFRAMES = int((RATE * RECORD_SECONDS) / CHUNK)
    c=1
    # initialize pyaudio
    p = pyaudio.PyAudio()
    getInputDevice(p)

    print('opening stream...')
    try:
        #print('first attempt')
        stream = p.open(format = FORMAT,
                        channels = CHANNELS,
                        rate = RATE,
                        input = True,
                        frames_per_buffer = CHUNK,
                        input_device_index = 1)
    except:
        #print('second attempt')
        stream = p.open(format = FORMAT,
                        channels = CHANNELS,
                        rate = RATE,
                        input = True,
                        frames_per_buffer = CHUNK,
                        input_device_index = 0)

    # discard first 1 second
    for i in range(0, NFRAMES):
        data = stream.read(CHUNK, exception_on_overflow = False)

    try:
        while True:
            print("Listening...")
            #disp.show_txt(0, 0, "Listening...", False)
            #showToScreen("Listening")
            
            frames = []
            for i in range(0, NFRAMES):
                data = stream.read(CHUNK, exception_on_overflow = False)
                frames.append(data)

            # process data
            # 4096 * 3 frames * 2 channels * 4 bytes = 98304 bytes 
            # CHUNK * NFRAMES * 2 * 4 
            buffer = b''.join(frames)
            audio_data = np.frombuffer(buffer, dtype=np.int32)
            nbytes = CHUNK * NFRAMES 
            # reshape for input 
            audio_data = audio_data.reshape((nbytes, 2))
            # run inference on audio data 
            pred=run_inference(disp, audio_data)
            
            if (pred=="silent"):
                time.sleep(SLEEP_TIME)
                PREDICTIONS.clear()
                continue
            
            PREDICTIONS.append(pred)
            if(len(PREDICTIONS)>=2):
                
                print(PREDICTIONS)
                p1=PREDICTIONS[0]
                p2=PREDICTIONS[1]
                        
                if(p1==p2):
                    
                    #we are certain it predicted the sound correctly
                    #send a notification to the smart home resident device using mqtt
                    if(NOR>0):
                    	#a person is in the room, so ignore
                        print("ignore because there are ", NOR ,"person(s) in the room")
                    else:
                    	#nobody is in the room. hence simulate and send notification to residents phone
                        print("sending a notification to the residents phone to turn off the ",pred)
                    PREDICTIONS.clear()
                    time.sleep(SLEEP_TIME)
            
            # save audio file
            #wavfile.write(WAVE_OUTPUT_FILENAME+str(c), RATE, audio_data)
            #c=c+1
    except KeyboardInterrupt:
        print("exiting...")
           
    stream.stop_stream()
    stream.close()
    p.terminate()

def process_audio_data(waveform):
    """Process audio input.

    This function takes in raw audio data from a WAV file and does scaling 
    and padding to 16000 length.

    """

    if VERBOSE_DEBUG:
        print("waveform:", waveform.shape, waveform.
        dtype, type(waveform))
        print(waveform[:5])

    # if stereo, pick the left channel
    if len(waveform.shape) == 2:
        #print("Stereo detected. Picking one channel.")
        waveform = waveform.T[1]
    else: 
        waveform = waveform 

    if VERBOSE_DEBUG:
        print("After scaling:")
        print("waveform:", waveform.shape, waveform.dtype, type(waveform))
        print(waveform[:5])

    # normalise audio
    wabs = np.abs(waveform)
    wmax = np.max(wabs)
    waveform = waveform / wmax

    PTP = np.ptp(waveform)
    #print("peak-to-peak: %.4f. Adjust as needed." % (PTP,))

    # return None if too silent 
    if PTP < 0.2:
        return []

    if VERBOSE_DEBUG:
        print("After normalisation:")
        print("waveform:", waveform.shape, waveform.dtype, type(waveform))
        print(waveform[:5])

    # scale and center
    waveform = 2.0*(waveform - np.min(waveform))/PTP - 1

    # extract 16000 len (1 second) of data   
    max_index = np.argmax(waveform)  
    start_index = max(0, max_index-8000)
    end_index = min(max_index+8000, waveform.shape[0])
    waveform = waveform[start_index:end_index]

    # Padding for files with less than 16000 samples
    if VERBOSE_DEBUG:
        print("After padding:")

    waveform_padded = np.zeros((16000,))
    waveform_padded[:waveform.shape[0]] = waveform

    if VERBOSE_DEBUG:
        print("waveform_padded:", waveform_padded.shape, waveform_padded.dtype, type(waveform_padded))
        print(waveform_padded[:5])

    return waveform_padded

def get_spectrogram(waveform):
    
    waveform_padded = process_audio_data(waveform)

    if not len(waveform_padded):
        return []

    # compute spectrogram 
    f, t, Zxx = signal.stft(waveform_padded, fs=16000, nperseg=255, 
        noverlap = 124, nfft=256)
    # Output is complex, so take abs value
    spectrogram = np.abs(Zxx)

    if VERBOSE_DEBUG:
        print("spectrogram:", spectrogram.shape, type(spectrogram))
        print(spectrogram[0, 0])
        
    return spectrogram

def get_spectrogram0(waveform):
    # Zero-padding for an audio waveform with less than 16,000 samples.
    waveform_padded = process_audio_data(waveform)

    if not len(waveform_padded):
        return []

    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = signal.stft(
        waveform_padded, frame_length=255, frame_step=128)
    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)
    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram
    
    
def run_inference(disp, waveform):

    # get spectrogram data 
    spectrogram = get_spectrogram(waveform)
    #print('len of spectrogram', len(spectrogram))
    if not len(spectrogram):
        #disp.show_txt(0, 0, "Silent. Skip...", True)
        showToScreen("Silent. Skip...")
        print("Too silent. Skipping...")
        #time.sleep(1)+
        return "silent"

    spectrogram1= np.reshape(spectrogram, (-1, spectrogram.shape[0], spectrogram.shape[1], 1))
    
    if VERBOSE_DEBUG:
        print("spectrogram1: %s, %s, %s" % (type(spectrogram1), spectrogram1.dtype, spectrogram1.shape))

    # load TF Lite model
    #interpreter = Interpreter('simple_audio_model_numpy.tflite')
    #interpreter = Interpreter('dash_home.tflite')
    #interpreter = Interpreter('dash_numpy.tflite')
    interpreter = Interpreter('dash_audio_84acc.tflite')
    #interpreter = Interpreter('dash_audio_93acc.tflite')
    #interpreter = Interpreter('dash_audio_94acc.tflite')
    #interpreter = Interpreter('dash_audio_73acc.tflite')
    interpreter.allocate_tensors()
    
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    #print(input_details)
    #print(output_details)

    input_shape = input_details[0]['shape']
    input_data = spectrogram1.astype(np.float32)
    #print(input_data.shape)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    print("running inference...")
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    yvals = output_data[0]
    #commands = ['Heater', 'Fan', 'AC', 'TV'] #dash_numpy
    commands = ['Heater', 'Fan',  'TV','AC'] #84_acc
    #commands = [ 'AC', 'Fan', 'Heater', 'TV'] #73,93,94_acc

    if VERBOSE_DEBUG:
        print(output_data[0])
    print(">>> " + commands[np.argmax(output_data[0])].upper())
    prediction=commands[np.argmax(output_data[0])].upper()
    showToScreen(prediction)
    #disp.show_txt(0, 12, commands[np.argmax(output_data[0])].upper(), True)
    
    #time.sleep(1)
    return prediction
def main():

    # create parser
    descStr = """
    This program does ML inference on audio data.
    """
    parser = argparse.ArgumentParser(description=descStr)
    # add a mutually exclusive group of arguments
    group = parser.add_mutually_exclusive_group()

    # add expected arguments
    group .add_argument('--input', dest='wavfile_name', required=False)
    
    # parse args
    args = parser.parse_args()

    disp = 1#SSD1306_Display()
    
    # test WAV file
    if args.wavfile_name:
        wavfile_name = args.wavfile_name
        # get audio data 
        rate, waveform = wavfile.read(wavfile_name)
        # run inference
        run_inference(disp, waveform)
    else:
        get_live_input(disp)

    print("done.")

# main method
if __name__ == '__main__':
    # creating thread
    t1 = threading.Thread(target=pir_sensing)
    t2 = threading.Thread(target=main)
 
    # starting thread 1
    t1.start()
    # starting thread 2
    t2.start()
 
    # wait until thread 1 is completely executed
    t1.join()
    # wait until thread 2 is completely executed
    t2.join()
 
    # both threads completely executed
    print("Done!")