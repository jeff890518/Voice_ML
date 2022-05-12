import wave
import librosa
import scipy.misc
import numpy as np
import pyaudio
import random, string
import tensorflow as tf
import vad_doa as vad

# 获取网络模型
model = tf.keras.models.load_model('ASR.h5')

# 录音参数
RESPEAKER_RATE = 16000
RESPEAKER_CHANNELS = 4 
RESPEAKER_WIDTH = 2
# run getDeviceInfo.py to get index
RESPEAKER_INDEX = 2  # refer to input device id
CHUNK = 1024
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "output.wav"

# 打开录音
p = pyaudio.PyAudio()

stream = p.open(
            rate=RESPEAKER_RATE,
            format=p.get_format_from_width(RESPEAKER_WIDTH),
            channels=RESPEAKER_CHANNELS,
            input=True,
            input_device_index=RESPEAKER_INDEX,)

def wav2mfcc(file_path, max_pad_len=160):
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    wave = wave[::2]
    mfcc = librosa.feature.mfcc(wave, sr=16000)
    pad_width = max_pad_len - mfcc.shape[1]
    mfcc = np.pad(mfcc, pad_width=((90, 90), (0, pad_width)), mode='constant')
    return mfcc
    
def save_wav_channel(fn, wav, channel):
    '''
    Take Wave_read object as an input and save one of its
    channels into a separate .wav file.
    '''
    # Read data
    nch   = wav.getnchannels()
    depth = wav.getsampwidth()
    wav.setpos(0)
    sdata = wav.readframes(wav.getnframes())

    # Extract channel data (24-bit data not supported)
    typ = { 1: np.uint8, 2: np.uint16, 4: np.uint32 }.get(depth)
    if not typ:
        raise ValueError("sample width {} not supported".format(depth))
    if channel >= nch:
        raise ValueError("cannot extract channel {} out of {}".format(channel+1, nch))
    print ("Extracting channel {} out of {} channels, {}-bit depth".format(channel+1, nch, depth*8))
    data = np.fromstring(sdata, dtype=typ)
    ch_data = data[channel::nch]

    # Save channel to a separate file
    outwav = wave.open(fn, 'w')
    outwav.setparams(wav.getparams())
    outwav.setnchannels(1)
    outwav.writeframes(ch_data.tostring())
    outwav.close()
# 读取音频数据
def load_data(data_path):
    mfcc = wav2mfcc(data_path)
    mfcc_reshaped = mfcc.reshape(1, 200, 160)
    return mfcc_reshaped

# 获取录音数据
def record_audio():
    print("* recording")

    frames = []

    for i in range(0, int(RESPEAKER_RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(RESPEAKER_CHANNELS)
    wf.setsampwidth(p.get_sample_size(p.get_format_from_width(RESPEAKER_WIDTH)))
    wf.setframerate(RESPEAKER_RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    return WAVE_OUTPUT_FILENAME


# 预测
def infer(audio_data):
    result = model.predict(audio_data)
    # lab = tf.argmax(result, 1)
    return result


if __name__ == '__main__':
    try:
        while True:
            # 加载数据
            data = load_data(record_audio())

            # 获取预测结果
            label = infer(data)
            if (label[0][0] < 0.5):
                angle = vad.vad()
                if 20<angle<=70:
                    pred='about northeast has emergency vehicle'
                if 70<angle<=110:
                    pred='about east has emergency vehicle'
                if 110<angle<=160:
                    pred='about southeast has emergency vehicle'
                if 160<angle<=200:
                    pred='about south has emergency vehicle'
                if 200<angle<=250:
                    pred='about southwest has emergency vehicle'
                if 250<angle<=290:
                    pred='about west has emergency vehicle'
                if 290<angle<=340:
                    pred='about northwest has emergency vehicle'
                if angle>340 or angle<=20:
                    pred='about north has emergency vehicle'
            else:
                pred='non emergency vehicle' 
            print("Prediction:",pred)
    except Exception as e:
        print(e)
        stream.stop_stream()
        stream.close()
        p.terminate()
