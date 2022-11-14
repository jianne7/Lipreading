import wave
import random
import array
import math
import numpy as np
import glob
from pydub import AudioSegment
from tqdm.notebook import tqdm
import torch
import numpy as np
import random
import os


class NoiseMixer:
    def __init__(self, song_folder:str, noise_folder:str, output_path:str = '.',
                 aug_num:int = 1):
        self.cleans = len(glob.glob(f'{song_folder}/*.wav'))
        self.song_list = self.sr16000(glob.glob(f'{song_folder}/*.wav'))
        self.noise_list = self.mp3_to_wav(glob.glob(f'{noise_folder}/*.mp3'))
        self.output_path = output_path
        self.aug_num = aug_num
    
    def __call__(self):
        for _ in range(self.aug_num):
            song_idx = random.randint(0, len(self.song_list)-1)
            noise_idx = random.randint(0, len(self.noise_list)-1)
            noise_wav = wave.open(self.noise_list[noise_idx], 'r') # generator
            clean_wav = wave.open(self.song_list[song_idx], 'r')
            
            # sr test
            noise_sr = noise_wav.getframerate()
            clean_sr = clean_wav.getframerate()
            print('noise :', noise_sr, 'clean :', clean_sr)
            assert noise_wav.getframerate() == 16000, 'noise sr is wrong'
            assert clean_wav.getframerate() == 16000, 'clean sr is wrong'
            
            snr = np.random.choice([2,3,4,5,6], p=[0.1,0.2,0.2,0.2,0.3])
            clean_amp = self.cal_amp(clean_wav)
            noise_amp = self.cal_amp(noise_wav)
            
            # 시작 지점부터 clean_amp 더한 값이 최소한 noise_amp 길이를 초과하면 안 되므로 
            start = random.randint(0, len(noise_amp)-len(clean_amp))
            split_noise_amp = noise_amp[start: start + len(clean_amp)]
            clean_rms = self.cal_rms(clean_amp)
            noise_rms = self.cal_rms(split_noise_amp)
            adjusted_noise_rms = self.cal_adjusted_rms(clean_rms, snr)
            adjusted_noise_amp = split_noise_amp * (adjusted_noise_rms / noise_rms)
            mixed_amp = (clean_amp + adjusted_noise_amp)
            # 16bit 넘을 경우 정규화
            if (mixed_amp.max(axis=0) > 32767):
                    mixed_amp = mixed_amp * (32767/mixed_amp.max(axis=0))
                    clean_amp = clean_amp * (32767/mixed_amp.max(axis=0))
                    adjusted_noise_amp = adjusted_noise_amp * (32767/mixed_amp.max(axis=0))
            name = self.song_list[song_idx].split('/')[-1]
            noisy_wave = wave.Wave_write(f'{self.output_path}/noisy_{_}_{name}')
            noisy_wave.setparams(clean_wav.getparams())
            noisy_wave.writeframes(array.array('h', mixed_amp.astype(np.int16)).tostring() )
            noisy_wave.close()
    
    def mp3_to_wav(self, noise_list):
        for idx, noise in enumerate(noise_list):
            mp3 = AudioSegment.from_mp3(noise)
            mp3 = mp3.set_frame_rate(16000)
            rename = noise.split('.')[:-1][0]
            mp3.export(f'{rename}.wav', format='wav')
        path = '/'.join(noise.split('/')[:-1])
        return glob.glob(f'{path}/*.wav')
    
    def sr16000(self, clean_list):
        for idx, clean in enumerate(clean_list):
            cleaning = AudioSegment.from_wav(clean)
            cleaning = cleaning.set_frame_rate(16000)
            rename = clean.split('.')[:-1][0]
            cleaning.export(f'{rename}_sr.wav', format='wav')
        path = '/'.join(clean.split('/')[:-1])
        return glob.glob(f'{path}/*_sr.wav')
    
    def cal_amp(self, wav_generator):
        buffer = wav_generator.readframes(wav_generator.getnframes())
        amptitude = (np.frombuffer(buffer, dtype="int16")).astype(np.float64)
        return amptitude

    def cal_rms(self, amp):
        return np.sqrt(np.mean(np.square(amp), axis=-1))

    def cal_adjusted_rms(self, clean_rms, snr):
        a = float(snr) / 20
        noise_rms = clean_rms / (10**a)
        return noise_rms
    
    
if __name__=='__main__':
    # for reproducibility
    seed = 1051

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    song_list = '/home/ubuntu/nia/Final_Test/data/Train/Train/Audio'
    noise_list = '/home/ubuntu/nia/Final_Test/data/Noise/'
    output_path = '/home/ubuntu/nia/Final_Test/data/Noisy'
    Mixing_num = len(os.listdir(song_list)) * 30
    noisy = NoiseMixer(song_list, noise_list, output_path, Mixing_num)
    noisy()