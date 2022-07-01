from flask import Flask, request, jsonify

import tkinter as tk
from tkinter import ttk, filedialog as fd, StringVar
from tkinter import *
import os
from tkinter import filedialog
import random
from cv2 import threshold
from matplotlib import text
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sound import load_audio
from numpy.typing import *
from sound.level import calculate_rms, rms2db, calculate_lamax, calculate_laeq, find_spike_indices, lamax_adjustment
from sound.weighting import A_weighting
import librosa
import math
import numpy as np
from scipy import signal
FORMAT = "utf-8"
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# 설문조사 결과에 따른 sensitivity를 반영하는 함수 필요
# 구현된 바 없으므로 default값으로 normal 설정
sensitivity = "normal"

# 각 지표의 default값 선언
user_laeq = ""
user_lamax = ""
user_threshold = ""
user_noise_level = ""
user_sensitivity = sensitivity
user_l_index = np.nan

# 모바일 애플리케이션 내 파일 선택/녹음 기능으로 filename 설정
filename = 'trial.wav'

laeq_range_beg_str = ""
laeq_range_end_str = ""
laeq_str = ""
lamax_str = ""
threshold_str = ""
exceed_count_str = ""

audio_info = {
    'generator': iter([]),
    'wav': np.array([]),
    'sr': 0,
    'spl': np.array([]),
    'len': 1.0,
    'frame_start_t': 0.0,
    'processed_t': 0.0,
}

analysis_info = {
    'beg': 0.0,
    'end': 0.0,
    'threshold': 57.0,
    'spike_indices': np.array([], dtype=np.int32),
}

def open_audio_file():
    filetypes = (
        ('raw audio file', '.wav'),
    )
    filename = fd.askopenfilename(filetypes=filetypes)
    if filename != '':
        stream, sr = load_audio(filename, block_len=220)
        audio_info['generator'] = stream
        audio_info['sr'] = sr
        audio_info['processed_t'] = 0.0
        audio_info['frame_start_t'] = 0.0

        next_audio_frame()
        
def fileset():
    stream, sr = load_audio(filename, block_len=220)
    audio_info['generator'] = stream
    audio_info['sr'] = sr
    audio_info['processed_t'] = 0.0
    audio_info['frame_start_t'] = 0.0

    next_audio_frame()
        
def next_audio_frame():
    wav = next(audio_info['generator'], None)
    if wav is None:
        return
    wav, sr = A_weighting(wav, audio_info['sr'])
    rms = calculate_rms(wav, sr)
    spl = rms2db(rms)

    audio_info['wav'] = wav
    audio_info['spl'] = spl[0]
    audio_info['len'] = librosa.get_duration(y=wav, sr=sr)
    audio_info['frame_start_t'] = audio_info['processed_t']
    audio_info['processed_t'] += audio_info['len']

    laeq_range_beg_str = f"{audio_info['frame_start_t']:.7f}"
    laeq_range_end_str = f"{audio_info['processed_t']:.7f}"

    update_laeq_lamax_label()
    update_threshold()
    
def std_threshold(sensitivity):
    if sensitivity == 'low':
        return 61.5
    elif sensitivity == 'high':
        return 56.0
    else:
        return 57.0
    
def update_laeq_lamax_label():
    global laeq_range_beg_str
    global laeq_range_end_str
    spl_len = audio_info['spl'].shape[0]
    if spl_len == 0:
        return
    start_t = audio_info['frame_start_t']
    try:
        # beg_sec = float(laeq_range_beg_str.get())
        beg_sec = float(laeq_range_beg_str)
    except ValueError:
        beg_sec = start_t
    try:
        # end_sec = float(laeq_range_end_str.get())
        end_sec = float(laeq_range_end_str)
    except ValueError:
        end_sec = audio_info['len']
        laeq_range_end_str = f"{audio_info['len']}"
    beg_sec = np.clip(beg_sec, start_t, audio_info['processed_t'])
    end_sec = np.clip(end_sec, start_t, audio_info['processed_t'])
    beg = round(spl_len * ((beg_sec - start_t) / audio_info['len']))
    end = round(spl_len * ((end_sec - start_t) / audio_info['len']))
    if beg > end:
        beg, end = end, beg
        beg_sec, end_sec = end_sec, beg_sec
    elif beg == end:
        end += 1
    beg_sec = (beg / spl_len) * audio_info['len'] + start_t
    end_sec = (end / spl_len) * audio_info['len'] + start_t
    laeq = calculate_laeq(audio_info['spl'], frames_beg=beg, frames_end=end)
    lamax = calculate_lamax(audio_info['spl'], frames_beg=beg, frames_end=end)
    analysis_info['beg'] = beg_sec
    analysis_info['end'] = end_sec
    
    return laeq, lamax

def update_threshold():
    global threshold_str
    spl_len = audio_info['spl'].shape[0]
    if spl_len == 0:
        return
    try:
        analysis_info['threshold'] = float(threshold_str)
    except ValueError:
        analysis_info['threshold'] = std_threshold(sensitivity)
        threshold_str = f"{analysis_info['threshold']}"

    analysis_info['spike_indices'] = find_spike_indices(audio_info['spl'], analysis_info['threshold'])
    exceed_count_str = f"Exceed count: {analysis_info['spike_indices'].shape[0]}"
    
    return analysis_info['spike_indices'].shape[0]

def update_noise_level(user_sensitivity, user_lamax):
    if user_sensitivity == 'low':
        if user_lamax <= 40:
            return 0
        elif user_lamax <= 43:
            return 1
        elif user_lamax <= 50:
            return 2
        elif user_lamax <= 58:
            return 3
        elif user_lamax <= 65:
            return 4
        elif user_lamax <= 76:
            return 5
        else:
            return 6
    elif user_sensitivity == 'high':
        if user_lamax <= 39:
            return 0
        elif user_lamax <= 41:
            return 1
        elif user_lamax <= 47:
            return 2
        elif user_lamax <= 53.5:
            return 3
        elif user_lamax <= 58.5:
            return 4
        elif user_lamax <= 68:
            return 5
        else:
            return 6
    else: #user_sensitivity == 'normal
        if user_lamax <= 40:
            return 0
        elif user_lamax <= 42.5:
            return 1
        elif user_lamax <= 48:
            return 2
        elif user_lamax <= 54:
            return 3
        elif user_lamax <= 60:
            return 4
        elif user_lamax <= 73:
            return 5
        else:
            return 6        
    
def update_lindex(filename):
    wav, sr = librosa.load(filename)
    wav, sr = A_weighting(wav, sr)
    ref = 2 * 10 ** (-5)

    f63 = signal.firwin(101, cutoff=[62, 64], fs=sr, pass_zero='bandpass')
    f125 = signal.firwin(101, cutoff=[122, 124], fs=sr, pass_zero='bandpass')
    f250 = signal.firwin(101, cutoff=[249, 251], fs=sr, pass_zero='bandpass')
    f500 = signal.firwin(101, cutoff=[499, 501], fs=sr, pass_zero='bandpass')
    
    band63 = signal.lfilter(f63, [1,0], wav)
    band125 = signal.lfilter(f125, [1,0], wav)
    band250 = signal.lfilter(f250, [1,0], wav)
    band500 = signal.lfilter(f500, [1,0], wav)
    
    s63, phase63 = librosa.magphase(librosa.stft(band63, center=False))
    s125, phase125 = librosa.magphase(librosa.stft(band125, center=False))
    s250, phase250 = librosa.magphase(librosa.stft(band250, center=False))
    s500, phase500 = librosa.magphase(librosa.stft(band500, center=False))
    
    rms63 = librosa.feature.rms(S=s63)
    rms125 = librosa.feature.rms(S=s125)
    rms250 = librosa.feature.rms(S=s250)
    rms500 = librosa.feature.rms(S=s500)
    
    spl63 = 10 * np.log10((rms63 ** 2) / (ref ** 2)) + 10.5006
    spl125 = 10 * np.log10((rms125 ** 2) / (ref ** 2)) + 10.5006
    spl250 = 10 * np.log10((rms250 ** 2) / (ref ** 2)) + 10.5006
    spl500 = 10 * np.log10((rms500 ** 2) / (ref ** 2)) + 10.5006
    
    band_lamax = [
        np.max(spl63).item(),
        np.max(spl125).item(),
        np.max(spl250).item(),
        np.max(spl500).item()
    ]
    
    if band_lamax[0]>110 or band_lamax[1]>100 or band_lamax[2]>93 or band_lamax[3]>87:
        l_index = 85
    elif band_lamax[0]>105 or band_lamax[1]>95 or band_lamax[2]>88 or band_lamax[3]>82:
        l_index = 80
    elif band_lamax[0]>100 or band_lamax[1]>90 or band_lamax[2]>83 or band_lamax[3]>77:
        l_index = 75
    elif band_lamax[0]>95 or band_lamax[1]>85 or band_lamax[2]>78 or band_lamax[3]>72:
        l_index = 70
    elif band_lamax[0]>90 or band_lamax[1]>80 or band_lamax[2]>73 or band_lamax[3]>67:
        l_index = 65
    elif band_lamax[0]>85 or band_lamax[1]>75 or band_lamax[2]>68 or band_lamax[3]>62:
        l_index = 60
    elif band_lamax[0]>80 or band_lamax[1]>70 or band_lamax[2]>63 or band_lamax[3]>57:
        l_index = 55
    elif band_lamax[0]>75 or band_lamax[1]>65 or band_lamax[2]>58 or band_lamax[3]>52:
        l_index = 50
    elif band_lamax[0]>70 or band_lamax[1]>60 or band_lamax[2]>53 or band_lamax[3]>47:
        l_index = 45
    elif band_lamax[0]>65 or band_lamax[1]>55 or band_lamax[2]>48 or band_lamax[3]>42:
        l_index = 40
    elif band_lamax[0]>60 or band_lamax[1]>50 or band_lamax[2]>43 or band_lamax[3]>37:
        l_index = 35
    else:
        l_index = 30
        
    return l_index

@app.route('/')
def setparameter():
    global user_laeq, user_lamax, user_threshold, user_noise_level, user_sensitivity, user_l_index
    fileset()
    user_laeq, user_lamax = update_laeq_lamax_label()
    user_threshold = update_threshold()
    user_noise_level = update_noise_level(user_sensitivity, user_lamax)
    user_l_index = update_lindex(filename)
    return "Parameters are set"

@app.route('/showresult')
def showparameter():
    return jsonify({
        "laeq": str(user_laeq),
        "lamax": str(user_lamax),
        "sensitivity": str(user_sensitivity),
        "threshold": str(user_threshold),
        "noise_level": str(user_noise_level),
        "l-Index" : str(user_l_index)
    })
 
if __name__ == "__main__":
    app.run()