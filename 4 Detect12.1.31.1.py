import argparse
import contextlib
import datetime as dt
import time as tm
import io

from io import BytesIO

import keyboard

import librosa, librosa.display
import numpy as np
import os

from multiprocessing import Process, Queue

#import pygame
#from pygame import mixer
import pyaudio

from scipy.io import wavfile as wav
from scipy.io.wavfile import read, write
from scipy.signal import freqs, sosfilt, find_peaks, peak_prominences
from scipy import stats
from sklearn import preprocessing
import sounddevice as sd
import soundfile as sf
import sys

import threading
import traceback

import warnings
import wave



form_1 = pyaudio.paInt16 # 16-bit resolution
chans = 1 # 1 channel
samp_rate = 96000 #44100 # 44.1kHz sampling rate-+c
chunk = 4096 # 2^12 samples for buffer
record_secs = 5 # seconds to record
DURATION = record_secs
dev_index = 1 #2 # device index found by p.get_device_info_by_index(ii)
FRAME_SIZE = 2048
HOP_SIZE = 512
OFFSET = 0

lq = Queue()
pq = Queue()

bufferio = io.BytesIO()

rec_samp = []  #moved to top level


warnings.simplefilter(action='ignore', category = FutureWarning)


def spike_funct(sf_signal):
    peaks, _ = find_peaks(sf_signal, height=0, distance = 3000)

    if len(peaks) > 1:
        sf_diffs = np.diff(sf_signal[peaks])
        return (np.min(sf_diffs), np.max(sf_diffs))
    else:
        return(-999,999)



def is_moving_funct(im_signal, im_duration):
    is_moving = "      Signal appears realtively stationary" #initial state
    rms_1 = librosa.feature.rms(y=im_signal,S = None , frame_length = FRAME_SIZE,
                                hop_length = HOP_SIZE)[0]
    max_amp_arr = np.array([])
    rms_factor = int((len(rms_1) / im_duration) * 0.25) #check rms every half second

    sig_to_a_rms_ratio = np.max(im_signal)/np.average(rms_1)

    for g in range(0,int(im_duration*2)):
        max_in_g = np.max(rms_1[g*rms_factor:(g+1)*rms_factor])
        ave_in_g = np.average(rms_1[g*rms_factor:(g+1)*rms_factor])
        max_amp_arr = np.append(max_amp_arr, max_in_g)

    prev_sign = "+"
    peak_counter = 0


    max_amp_diff = np.diff(max_amp_arr)

    #How to detect a sign change for elements in a numpy array
    #answer by janneb
    #https://stackoverflow.com/questions/2652368/how-to-detect-a-sign-change-for-elements-in-a-numpy-array
    amp_sign = np.sign(max_amp_diff)
    amp_change = np.count_nonzero(((np.roll(amp_sign, 1) - amp_sign) != 0).astype(int))

    if amp_change == 0:   #appears to be approaching microphone
        is_moving = "       Signal appears to be growing or fading"
    elif 1 <= amp_change <= 2: #appears to be passing microphone
                is_moving = "       Signal appears to be growing then fading"

    return(is_moving)



def flat_funct(ff_signal):
    ff_flatness = librosa.feature.spectral_flatness(y=ff_signal)
    ff_flat_ave = np.average(ff_flatness)
    ff_flat_diff = np.diff(ff_flatness)
    ff_flat_diff_abs = abs(ff_flat_diff)
    #ff_flat_count = np.count_nonzero(ff_flat_diff_abs>0.001) #original, not significant enough
    ff_flat_count2 = np.count_nonzero(ff_flat_diff_abs>0.0035)
    ff_flat_count3 = np.count_nonzero((ff_flat_diff_abs>0.001) & (ff_flat_diff_abs<0.0035))
    #print(ff_flat_ave,ff_flat_count2, ff_flat_count3)
    return(ff_flat_ave, ff_flat_count2, ff_flat_count3)


def sband_funct(sb_signal):
    #Spectral bandwidth
    sb_spec_bw = librosa.feature.spectral_bandwidth(y=sb_signal, sr=96000)
    peaks, _ = find_peaks(sb_spec_bw[0])
    prominences = peak_prominences(sb_spec_bw[0], peaks)[0]
    prom_ave = np.average(prominences)
    sb_sbw_min = np.min(sb_spec_bw)
    sb_sbw_max = np.max(sb_spec_bw)
   
    return(sb_sbw_min, sb_sbw_max, prom_ave)


def ae_funct(ae_funct_signal):
    FRAME_SIZE = 1024
    HOP_LENGTH = 128
    #ae_signal = amplitude_envelope(ae_funct_signal, FRAME_SIZE, HOP_LENGTH)
    
    #Velardo Calculating the amplitude envelope
    ae_signal = np.array([max(ae_funct_signal[i:i+FRAME_SIZE]) for i in range(0, ae_funct_signal.size,
                                                                              HOP_LENGTH)])
    
    ae_diff = np.diff(ae_signal)

    #How to detect a sign change for elements in a numpy array
    #answer by janneb
    #https://stackoverflow.com/questions/2652368/how-to-detect-a-sign-change-for-elements-in-a-numpy-array
    ae_sign = np.sign(ae_diff)
    ae_change = np.count_nonzero(((np.roll(ae_sign, 1) - ae_sign) != 0).astype(int))
    ae_diff_count = np.count_nonzero((ae_diff)>0.0005)
    
    #print(np.average(ae_signal), ae_change)
    return(np.average(ae_signal), ae_change, ae_diff_count)


#Audio data analysis using python Shubham Kumar Shukla
def normalize(x, axis = 0):
  return preprocessing.minmax_scale(x,  axis = axis)


def ro_test(ro_frames_lib):
    spectral_rolloff = librosa.feature.spectral_rolloff(ro_frames_lib + 0.01, 
            sr = samp_rate,roll_percent = 0.95)[0]

    n_sr = normalize(spectral_rolloff)
    ro_count = np.count_nonzero(np.where(n_sr > 0.8))

    sr_diff = np.diff(spectral_rolloff)
    sr_diff_ave = np.average(abs(sr_diff))
    
    #print(ro_counter, ro_name, sr_diff_ave, ro_count)

    #     plt.figure(figsize = (12, 4))
    #     librosa.display.waveshow(ro_frames_lib, sr = 96000, alpha = 0.4)
    #     plt.plot(librosa.times_like(spectral_rolloff, sr = 96000), 
    #        normalize(spectral_rolloff), color = 'r') #t, 
    #     plt.show()
    
    return(sr_diff_ave, ro_count)


def proc_data(): 
    pd_frames_lib = lq.get()
    ###

    put_var = 0
    g = 0
    pd_frames_lib = pd_frames_lib[1000:]                 #spike observed when starting audio
    
    ave_amp = np.average(abs(pd_frames_lib))
        
    print("   Test 1:  Average Amplitude")

    ave_amp_high = 0.01
    ave_amp_low = 0.0028
    
    if ave_amp_low <= ave_amp <= ave_amp_high:  #absolute amp range for objective signal
        print("      Average Amplitude within range: ", round(ave_amp,3))
        print("   Test 2:  Spike Test")
        
        max_down_diff, max_up_diff = spike_funct(pd_frames_lib)
        
        if max_down_diff > -0.035 and max_up_diff < 0.035:
            print("      Signal within tolerance")
            print("   Test 3:  Moving check")
            
            is_moving = is_moving_funct(pd_frames_lib, DURATION)
            
            if is_moving == "      Signal appears realtively stationary":
                print(is_moving)
                print("   Test 4:  Spectral Flatness")
               
                flat_ave, flat_count2, flat_count3 = flat_funct(pd_frames_lib)
                
                if (0.005 <= flat_ave <= 0.0098) and (1 <= flat_count2 <= 28) and (260 <= flat_count3 <= 450):
                    print("      Tonality coefficient average and variation fits saw profile:",
                          flat_count2, flat_count3)
                    print("   Test 5:  Amplitude Envelope")

                    ae_ave, ae_change, ae_diff_count = ae_funct(pd_frames_lib)
                    
                    if ((0.005<= ae_ave <= 0.06) and (1300 <= ae_change <= 1470) and 
                       (265 <= ae_diff_count <= 450)):
                        print("      Amplitude Envelope fits saw profile", ae_ave, 
                              ae_change, ae_diff_count)
                        print("   Test 6:  Spectral Bandwidth")
                        
                        sbw_min, sbw_max, sbw_prom_ave = sband_funct(pd_frames_lib)
                        
                        if (12000 <= sbw_min and sbw_max <= 15000 and (135 <= sbw_prom_ave <= 350) ):
                            print("      Spectral Bandwidth fits saw profile", 
                                  round(sbw_min,1), round(sbw_max,1), round(sbw_prom_ave,1))
                            print("   Test 7:  Rolloff Test")
                            
                            sr_diff_ave, sr_count = ro_test(pd_frames_lib)
                            
                            if  (165.0 <= sr_diff_ave <= 695.0) and (1 <= sr_count <= 20):
                                print("      Rolloff Test fits saw profile:",
                                      round(sr_diff_ave, 2), sr_count)
                                print("   Alarm")
                                put_var = 10
                            else:             
                                print("      Rolloff Test is outside range:",
                                      round(sr_diff_ave, 2), sr_count)      
                        else:
                            print("      Did not meet Spectral Bandwidth criteria:", 
                                  round(sbw_min,1), round(sbw_max,1), round(sbw_prom_ave,1))     
                    else:
                        print("      Did not meet Amplitude Envelope criteria:", ae_ave, 
                              ae_change, ae_diff_count)
                else:
                    print("      Tonality coefficient not within range:",flat_count2, flat_count3)
            else:
                print(is_moving)
        else:
            print("     Signal has uncharacteristic spikes in amplitude.", round(max_up_diff,3), round(max_down_diff,3)) 
            is_moving = "uncharacteristic spikes"
    elif ave_amp > ave_amp_high:
        print("      Average Amplitude ABOVE range: ", round(ave_amp,3))
        #is_moving = "Average Amplitude ABOVE range"
    elif ave_amp_low > ave_amp:  
        print("      Average Amplitude below range: ", round(ave_amp,3))
        #is_moving = "Average Amplitude below range"
        

    ###
    pq.put(put_var)


def write_alarm_wav(waw_rec_samp, waw_samp_rate):
    date = dt.datetime.now()  
    waw_file_name = ("Alarm_file" + "-" + str(date.year)+ "." + str(date.month)+ "." +
        str(date.day)+ "-" + str(date.hour)+ "." + str(date.minute)+"." +
        str(date.second) + ".wav")
    
    # save the audio frames as .wav file
    sf.write(waw_file_name, waw_rec_samp, waw_samp_rate)

    
def sound_alarm():
    #sounddevice sample code
    #print("sound alarm")
    data, fs = sf.read('Wake-up-sounds.wav', always_2d=True)
    sd.default.device = 0
    sd.default.samplerate = fs        
    while True:
        try:
            print("playing", fs)
            sd.play(data, 48000)#fs) #unable to change rate to 44100 in alsa.conf
            sd.wait()
            
        except KeyboardInterrupt:
            sys.exit('Interrupted by user')

            
def main():
    #inital state of veriables
    h = 0 #0 for first cycle, 1 for steady state, 10 for alarm
    prev_rec_samp = []
    prev_rec= []
    #pq.put(0) 
    sd.default.samplerate = samp_rate
    sd.default.channels = 1    
    sd.default.device = 1
    
    rec_samp = sd.rec(int(DURATION * samp_rate), samplerate = samp_rate, channels = chans,
                              device = 1)

    now = dt.datetime.now()
    print("Recording first sample:      ", str(now))
    
    sd.wait()
    
    prev_rec_samp = np.transpose(rec_samp,(1,0))
    
    while h <= 1:
        try:
            #process prev sample
            #record next sample
           
            if rec_samp.size > 0:     #don't process empty array
                p1 = Process(target=proc_data, ) #args=(, )
                #print("p1.start")
                p1.start()
                tp_rec_samp = np.transpose(rec_samp,(1,0))
                prev_rec_samp = rec_samp
                lq.put(tp_rec_samp[0])
                
            now = dt.datetime.now()
            print("Recording next sample at:    ", str(now))
            
            rec_samp = sd.rec(int(DURATION * samp_rate))

            if rec_samp.size > 0:      #empty array, nothing to join/close
                h = pq.get()
                p1.join()
                p1.close()
            elif h == 0:
                print("   First cycle, proc_data will start processing prev_data on second pass.")
                       
            if h == 10:                            #Alarm, while loop ends    
                #print("size", samp_rate)
                write_alarm_wav(prev_rec_samp , samp_rate) #prev_rec_samp, samp_rate)
                #print("pre sound alarm")
                sound_alarm()
                #print("post sound alarm")

            sd.wait()
            
        except KeyboardInterrupt:
            sys.exit("Ending Main")
            
        except Exception:
            traceback.print_exc()

if __name__ == "__main__":
    main() 

 