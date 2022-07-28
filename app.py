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
from sqlalchemy import true
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
    noise_level = 6
    t_low = (40, 43, 50, 58, 65, 76)
    t_normal = (40, 42.5, 48, 54, 60, 73)
    t_high = (39, 41, 47, 53.5, 58.5, 68)
    
    if user_sensitivity == 'low':
        for i in range(len(t_low)):
            if user_lamax <= t_low[i]:
                noise_level = i
    elif user_sensitivity == 'high':
        for i in range(len(t_high)):
            if user_lamax <= t_high[i]:
                noise_level = i
    else: # user_sensitivity == 'normal'
        for i in range(len(t_normal)):
            if user_lamax <= t_normal[i]:
                noise_level = i
    
    return noise_level
    
def update_lindex(filename):
    wav, sr = librosa.load(filename)
    wav, sr = A_weighting(wav, sr)
    ref = 2 * 10 ** (-5)

    f63 = signal.firwin(11, cutoff=[62, 64], fs=sr, pass_zero='bandpass')
    f125 = signal.firwin(11, cutoff=[122, 124], fs=sr, pass_zero='bandpass')
    f250 = signal.firwin(11, cutoff=[249, 251], fs=sr, pass_zero='bandpass')
    f500 = signal.firwin(11, cutoff=[499, 501], fs=sr, pass_zero='bandpass')
    
    # band63 = f63
    # band125 = f125
    # band250 = f250
    # band500 = f500
    
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

    
    l_index = 85
    t63 = tuple(range(110, 54, -5))
    t125 = tuple(range(100, 44, -5))
    t250 = tuple(range(93, 37, -5))
    t500 = tuple(range(87, 31, -5))
    t_index = tuple(range(85, 29, -5))
    
    for i in range(len(t_index)):
        if band_lamax[0] > t63[i] or band_lamax[1] > t125[i] or band_lamax[2] > t250[i] or band_lamax[3] > t500[i]:
            l_index = t_index[i]
            break
        
    return l_index

def analysis_sentence(user_noise_level, user_l_index):
    t_noise_level_sentences = [
        '실내 환경은 조용합니다. 외부 소음에 노출되지 않았습니다.',
        '소음이 당신을 괴롭히지 않습니다. 소음이 일상생활에 지장을 줄 만큼 크지 않습니다.',
        '인내심을 가지고 소음 문제에 대처할 수 있는 수준입니다. 소음이 지속되어 괴롭다면, 이웃과 소통하시기를 권장합니다.',
        '소음 문제에 대처하기 위해 이웃과 소통할 필요가 있습니다. 주의를 부탁드려 보세요. 이웃에게 측정 결과를 제시하고 이야기를 나눠보세요.',
        '소음이 당신의 휴식 시간을 방해합니다. 층간소음 법적 기준을 초과합니다. 이웃과 적극적으로 소통할 필요가 있습니다. 만약 소음이 계속 발생한다면 이웃사이 센터에 상담을 문의하세요.',
        '소음으로 인해 거주 공간에서 대화가 들리지 않습니다. 소음으로부터 견디기 어려운 상황입니다. 층간소음 법적 기준을 상당히 초과합니다. 이웃과 소통에 어려움이 있다면 이웃사이 센터에 상담을 문의하세요.',
        '사람이 이곳에 사는 것은 불가능합니다.'
    ]
    t_l_index_sentences = [
        ['뛰는 행위나 발걸음 소리가 거의 들리지 않습니다.\n', '의자나 물건이 떨어지는 소리가 전혀 들리지 않습니다.\n', '어린이가 크게 소리쳐도 괜찮습니다.\n'],
        ['뛰는 행위나 발걸음 소리가 조용할 때 들립니다.\n', '의자나 물건이 떨어지는 소리가 들리지 않습니다.\n', '다소 뛰어 다녀도 괜찮습니다.\n'],
        ['뛰는 행위나 발걸음 소리가 멀리서 들리는 느낌입니다.\n', '의자나 물건이 떨어지는 소리가 거의 들리지 않습니다.\n', '소음을 신경쓰지 않고 생활할 수 있습니다.\n'],
        ['뛰는 행위나 발걸음 소리가 들리지만 거슬리지 않습니다.\n', '샌들을 신고 걷는 소리가 들립니다.\n', '소음의 존재를 약간 알 수 있습니다.\n'],
        ['뛰는 행위나 발걸음 소리가 들리지만 거의 거슬리지 않습니다.\n', '칼을 사용하는 소리가 들립니다.\n', '소음을 약간 주의하면서 생활할 수 있습니다.\n'],
        ['뛰는 행위나 발걸음 소리에 약간의 마음이 쓰입니다.\n', '슬리퍼를 신고 걷는 소리가 들립니다.\n', '소음을 주의한다면 문제 없을 것입니다.\n'],
        ['뛰는 행위나 발걸음 소리가 약간 거슬립니다.\n', '수저를 떨어뜨리는 소리가 들립니다.\n', '소음이 상호 간에 견딜 수 있는 정도입니다.\n'],
        ['뛰는 행위나 발걸음 소리가 잘 들려 거슬립니다.\n', '동전을 떨어뜨리는 소리가 들립니다.\n', '소음을 발생시킨 어린이를 꾸짖게 됩니다.\n'],
        ['뛰는 행위나 발걸음 소리가 매우 잘 들려 거슬립니다.\n', '1원짜리 동전을 떨어뜨리는 소리가 들립니다.\n', '어른들도 마음을 쓰게 되는 수준입니다.\n'],
        ['뛰는 행위나 발걸음 소리가 매우 귀찮습니다.\n', '주의하여도 시비가 붙는 수준입니다.\n'],
        ['뛰는 행위나 발걸음 소리가 시끄러워 견딜 수 없습니다.\n', '소음을 참는 생활이 필요한 수준입니다.\n']
    ]
    
    l_index = list(range(30, 86, 5))
    
    sentence = ''
    sentence += '녹음된 전체 음원을 통해 주거 환경의 소음 발생 정도를 분석합니다. \n'
    for i in range(len(l_index)):
        if l_index[i] == user_l_index:
            for j in t_l_index_sentences[i]:
                sentence += j
    sentence += '발생한 최고 소음을 기반으로 층간소음의 정도를 분석합니다. \n'
    sentence += t_noise_level_sentences[user_noise_level]
    
    return sentence

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
def showresult():
    return jsonify({
        "laeq": str(user_laeq),
        "lamax": str(user_lamax),
        "sensitivity": str(user_sensitivity),
        "threshold": str(user_threshold),
        "noise_level": str(user_noise_level),
        "l-Index" : str(user_l_index)
    })
    
@app.route('/analysis')
def analysis():
    return analysis_sentence(user_noise_level, user_l_index)
    

# @app.route('/brief-analysis')
# def brief():
#     return
 
if __name__ == "__main__":
    app.run()