import numpy as np
from statsmodels.tsa.ar_model import AutoReg

# 0.5s window

def MAVori(data):
    return np.mean(np.abs(data))

def MAV(data,window_size,move):
    max=0
    for i in range((len(data)-window_size)//move+1):
        tmp=MAVori(data[i*move:(i*move+window_size)])
        if tmp>max:
            max=tmp
    return max

def WLori(data):
    return np.sum(np.abs(np.diff(data)))/len(data)

def WL(data,window_size,move):
    max=0
    for i in range((len(data)-window_size)//move+1):
        tmp=WLori(data[i*move:(i*move+window_size)])
        if tmp>max:
            max=tmp
    return max

def WAMPori(data):
    th_wamp=2
    s=np.abs(np.diff(data))
    s=[i for i in s if i>th_wamp]
    return np.size(s)

def WAMP(data,window_size,move):
    max=0
    for i in range((len(data)-window_size)//move+1):
        tmp=WAMPori(data[i*move:(i*move+window_size)])
        if tmp>max:
            max=tmp
    return max

def AR(data):
    p=4
    model_fit = AutoReg(data, lags=p).fit()
    return model_fit.params

def MAVSori(data):
    K=3
    MAV=np.zeros(K)
    MAVS=np.zeros(K-1)
    for i in range(K):
        MAV[i]=MAVori(data[i*16:(i+1)*16])
    for i in range(K-1):
        MAVS[i]=MAV[i+1]-MAV[i]
    return MAVS

def MAVS(data,window_size,move):
    max=np.zeros(2)
    for i in range((len(data)-window_size)//move+1):
        tmp=MAVSori(data[i*move:(i*move+window_size)])
        if tmp[0]>max[0] and tmp[1]>max[1]:
            max=tmp
    return max


def feature(data, window_size, move):
    max_mav_index = None
    max_mav_value = float('-inf')
    num_windows = (len(data) - window_size) // move + 1
    
    for i in range(num_windows):
        window_data = data[i * move:(i * move + window_size)]
        mav_value = MAVori(window_data)
        if mav_value > max_mav_value:
            max_mav_value = mav_value
            max_mav_index = i

    selected_window_data = data[max_mav_index * move:(max_mav_index * move + window_size)]

    wl_value = WLori(selected_window_data)
    wamp_value = WAMPori(selected_window_data)
    ar_value = AR(selected_window_data)
    mavs_value = MAVSori(selected_window_data)
    ret=np.array([max_mav_value, wl_value, wamp_value, *ar_value, *mavs_value])
    
    return ret


