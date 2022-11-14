import os
import glob
import numpy as np
import pandas as pd
import math
from tqdm import tqdm

import torch
import librosa
from torch.utils.data import Dataset, DataLoader

import jamotools


from fsn import *
from mp4tonpy import extract_opencv
from fps_changer import FpsChanger
from dataloader import *
from Label2id import *
from model.vpmodel import SpeechTransformer


def fps30_to_fps25(video25_path:str):
    if not os.path.isdir(video25_path):
        print('Changing fps...')
        os.mkdir(video25_path)
        fps_changer = FpsChanger(
                data_folder=video_path,
                fps_folder=video25_path,
                target_fps=25,
            )
        del fps_changer
    elif len(os.listdir(video25_path)) < 1:
        print('Changing fps...')
        fps_changer = FpsChanger(
            data_folder=video_path,
            fps_folder=video25_path,
            target_fps=25,
        )
        del fps_changer
    else:
        print('Finish Changing FPS')
        pass
            
def fps25_to_np(video_npy_path:str, video25_path, videos):
    if not os.path.isdir(video_npy_path):
        os.mkdir(video_npy_path)
        i=1
        for idx, video in enumerate(videos):
            data = extract_opencv(video) 
            path_to_save = os.path.join(video25_path.replace('Video_fps','Video_npy'),
                                        video.split('/')[-1][:-4]+'.npy')
            np.save(path_to_save, data)

    elif len(os.listdir(video_npy_path)) < 1:
        for idx, video in enumerate(videos):
            data = extract_opencv(video) 
            path_to_save = os.path.join(video25_path.replace('Video_fps','Video_npy'),
                                        video.split('/')[-1][:-4]+'.npy')
            np.save(path_to_save, data)
    else:
        print('Finish Changing NUMPY')
        pass
                
def noise_wav(audio_path, output_dir, fsn_checkpoint_path, device):
     # 오디오 노이즈 제거 프로세스 (DNN)
    NEG_INF = torch.finfo(torch.float32).min
    PI = math.pi
    SOUND_SPEED = 343  # m/s
    EPSILON = np.finfo(np.float32).eps
    MAX_INT16 = np.iinfo(np.int16).max
    dataset_dir_list = [audio_path,]
    
    if not os.path.isdir(output_dir):
        print('Changing denosing...')
        os.mkdir(output_dir)

        with torch.no_grad():
            fsn_model = Model(
                num_freqs=257,
                look_ahead=2,
                sequence_model="LSTM",
                fb_num_neighbors=0,
                sb_num_neighbors=15,
                fb_output_activate_function="ReLU",
                sb_output_activate_function=False,
                fb_model_hidden_size=512,
                sb_model_hidden_size=384,
                norm_type="offline_laplace_norm",
                num_groups_in_drop_band=2,
                weight_init=False,
            )
            
            infer = Inferencer(dataset_dir_list, fsn_checkpoint_path, output_dir)
            infer()
        print('Denoising model parameters : ', count_params(fsn_model))
    
    elif len(os.listdir(output_dir)) < 1:
        print('Changing denosing...')
  
        with torch.no_grad():
            fsn_model = Model(
                num_freqs=257,
                look_ahead=2,
                sequence_model="LSTM",
                fb_num_neighbors=0,
                sb_num_neighbors=15,
                fb_output_activate_function="ReLU",
                sb_output_activate_function=False,
                fb_model_hidden_size=512,
                sb_model_hidden_size=384,
                norm_type="offline_laplace_norm",
                num_groups_in_drop_band=2,
                weight_init=False,
            )

            infer = Inferencer(dataset_dir_list, fsn_checkpoint_path, output_dir)
            infer()
        print('Denoising model parameters : ', count_params(fsn_model))
    else:
        print('Finish DENOISING')        
        pass

                
def count_params(model):
    return sum(p.numel() for p in model.parameters())


class test_dataset(Dataset):
    def __init__(
            self,
            video_paths: list,              # list of video paths
            audio_paths: list,              # list of audio paths
            transcripts: list,              # list of transcript paths
            sos_id: int = 2,                    # identification of start of sequence token
            eos_id: int = 3,                    # identification of end of sequence token
            spec_augment: bool = False,     # flag indication whether to use spec-augmentation of not
            mode: str = 'train',
            ):
        super(test_dataset, self).__init__()
        
        # args : (sr, n_mels, frame_length, frame_shift)
        # 640 hop len == 16000 sr * 0.04 frame shift 
        self.transforms = FilterBank(16000, 80, 60, 40)
        self.video_paths = sorted(glob.glob(f'{video_paths}/*.npy'), key=lambda x: x.split('_')[-1]) 
        self.audio_paths = sorted(glob.glob(f'{audio_paths}/enhanced_0058/*.wav'), key=lambda x: x.split('_')[-2])
        
        # self.transcripts = sorted(glob.glob(transcripts + '/*.txt'), key=lambda x: x.split('_')[-1])
        self.dataset_size = len(self.audio_paths)
        
        if mode == 'train':
            self.preprocessing = Compose([RgbToGray(),
                                        Normalize( 0.0,255.0 ),
                                        CenterCrop((96, 96)),
                                        HorizontalFlip(0.5),
                                        Normalize(0.421, 0.165) ])
        else:
            self.preprocessing = Compose([RgbToGray(),
                                        Normalize( 0.0,255.0 ),
                                        CenterCrop((96, 96)),
                                        Normalize(0.421, 0.165) ])
        
        self.sos_id=sos_id
        self.eos_id=eos_id
        self.normalize = True # True
        
        self.VANILLA = 0
        # 나중에 증강된 파일만 확인하기 위해서 필요
        self.augment_existence = [self.VANILLA] * len(self.audio_paths)
        self.SPEC_AUGMENT = spec_augment    # SpecAugment
        self.spec_augment = SpecAugment(2, 2, [0, 40], [0, 30]) # config로 처리할것
        self.mode = mode
        
        # self._augment(spec_augment)
        # self.shuffle()


    def parse_audio(self, audio_path: str):
        # pdb.set_trace()
        signal, _ = librosa.load(audio_path, sr = 16000, mono=True) # wav, sr
        feature = self.transforms(signal)
        
        if self.mode == 'train' and self.SPEC_AUGMENT:
            feature = self.spec_augment(feature)

        if self.normalize:
            mean = torch.mean(feature, 0, keepdim=True)
            std = torch.std(feature, 0, keepdim=True)
            feature = (feature - mean) / (std + 1e-8)
        # print('audio', feature.shape)
        return feature
    
    def parse_video(self, video_path: str):
        video = np.load(video_path)
        # print('video', video.shape)
        video = self.preprocessing(video)
        # print('prepro', video.shape)
        video = torch.from_numpy(video).float()
        
        video -= torch.mean(video)
        video /= torch.std(video)
        video_feature  = video
        # video_feature = video_feature.permute(3,0,1,2) #T H W C --> C T H W
        
        return video_feature


    def __getitem__(self, index):
        if self.mode == 'train':
            video_feature = self.parse_video(self.video_paths[index])
            audio_feature = self.parse_audio(self.audio_paths[index])
            transcript = self.parse_transcript(self.transcripts[index])
            return video_feature, audio_feature, transcript
        elif self.mode == 'test':
            print(f'index:{index}, {self.video_paths[index]}, {self.audio_paths[index]}')
            video_feature = self.parse_video(self.video_paths[index])
            audio_feature = self.parse_audio(self.audio_paths[index])
            return video_feature, audio_feature

    def __len__(self):
        return len(self.audio_paths)

    def count(self):
        return len(self.audio_paths)

    
def collate_fn(batch):
    
    """ functions that pad to the maximum sequence length """
    batch = sorted(batch, key=lambda sample: sample[0].shape[0], reverse=True)
    video, vid_len, audio, aud_len = zip(*[(video, video.shape[0],
                                            audio, audio.shape[0])
                                            for (video, audio) in batch])
    # sort by sequence length for rnn.pack_padded_sequence()
    # pdb.set_trace()
    
    max_vid_sample = video[0]
    max_aud_sample = audio[0]

    max_vid_size = max(vid_len)
    max_aud_size = max(aud_len)
    
    vid_feat_x = max_vid_sample.size(1) # height
    vid_feat_y = max_vid_sample.size(2) # width
#     vid_feat_c = max_vid_sample.size(3) # channel
    aud_feat_f = max_aud_sample.size(1) # frequency
    batch_size = len(audio)

    vids = torch.zeros(batch_size, max_vid_size, vid_feat_x, vid_feat_y)
    auds = torch.zeros(batch_size, max_aud_size, aud_feat_f)

    # pdb.set_trace()
    for x in range(batch_size):
        sample = batch[x]
        video_ = sample[0]
        audio_ = sample[1]
        vid_len_ = video_.shape[0] # time steps
        aud_len_ = audio_.shape[0] # time steps
        vids[x, :vid_len_, :, :] = video_
        auds[x].narrow(0, 0, aud_len_).copy_(audio_)
    # B T H W -> B C T H W
    vids = vids.unsqueeze(1)
    return vids, auds

    
    
# def test(model, device, test_data_loader):
def test(model, test_data_loader):
    # torch.cuda.empty_cache()
    '''checkpoint filepath 수정 필요'''
    
    checkpoint = sorted(glob.glob('checkpoint/*.pt'))
    checkpoint = checkpoint[-1]
    
    device = torch.device('cpu')

    state = torch.load(checkpoint, map_location=torch.device('cpu'))
    model.load_state_dict(state['model'])
    model.to(device)
    
    # device = torch.device('cpu')
    model.eval()
    
    file_name = []
    submission = []
    
    with torch.no_grad():
        for i, (vid, aud) in enumerate(test_data_loader):

            t_a = aud.shape[1]
            t_v = vid.shape[2]

            if t_a != t_v:
                min_len = min(t_a, t_v)
                vid = vid[:, :, :min_len, :, :]  
                aud = aud[:, :min_len, :]

            vid = vid.to(device)
            aud = aud.to(device)

            token = model.search(aud, vid, max_length=127)

            txt = []
            for tok in token:
                txt.append(id2txt[tok])

            sentence = jamotools.join_jamos(txt[:-1])
            print(f'sentence:{sentence}')

            submission.append(sentence)
            file_name.append(f'{i+1:05d}')
    
    df = pd.DataFrame(zip(file_name, submission), columns=['file_name', 'answer'])
    df.to_csv('submission.csv', index=False, encoding='utf-8-sig')
    print('Save submission!')


if __name__ == "__main__":
    # for reproducibility
    seed = 1051
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    
    # 경로 지정
    video_path = "/home/ubuntu/nia/Final_Test/data/Test/Video"
    video25_path = "/home/ubuntu/nia/Final_Test/data/Test/Video_fps"
    video_npy_path = "/home/ubuntu/nia/Final_Test/data/Test/Video_npy"
    audio_path = "/home/ubuntu/nia/Final_Test/data/Test/Audio"
    fsn_checkpoint_path = '/home/ubuntu/nia/Final_Test/model/pretrained_weights/fullsubnet_best_model_58epochs.tar'
    output_dir = '/home/ubuntu/nia/Final_Test/data/Test/Denoise_wav'
    
    # 오름차순으로 파일 경로 정렬
    audios = sorted(glob.glob(f'{audio_path}/*.wav'))
    
    # 비디오 전처리
    fps30_to_fps25(video25_path)
    
    videos = sorted(glob.glob(f'{video25_path}/*.mp4'))

    fps25_to_np(video_npy_path, video25_path, videos)
    
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    
    # 오디오 전처리
    noise_wav(audio_path, output_dir, fsn_checkpoint_path, device)

    
    # vocab 로드
    txt2id, id2txt = vocab_generator()
    blank_id = txt2id['<blank>']
    
    # 멀티 모달 하이퍼 파라미터
    max_length = 256 
    batch_size = 1
    learning_rate = 5e-5 # 0.0015 # (espnet)
    max_vocab_size = 71
    dset_args = {'video_path' : video_npy_path,
                'audio_path' : output_dir,
                'transcript_path' : '/Non',
                'batch_size' : 1,
                'workers' : 0
                }
    vid_args = VideoArgs()
    vid_args.batch_size = 1
    
    # 멀티 모달
    model_device = torch.device('cpu')
    model = SpeechTransformer(vocab_size=max_vocab_size, device=model_device)
    model = model.to(model_device) # 수정필요
    print('Multi-Modal parameters : ', count_params(model))
    
    # 데이터셋 및 데이터로더
    dsets = test_dataset(
            video_paths=dset_args['video_path'],
            audio_paths=dset_args['audio_path'],
            transcripts=dset_args['transcript_path'],
            spec_augment=False,
            mode='test'
            )
    test_data_laoder = DataLoader(dsets,
                                batch_size=dset_args['batch_size'],
                                shuffle=False,
                                collate_fn=collate_fn,
                                pin_memory=False,
                                num_workers=dset_args['workers'],
                                worker_init_fn=np.random.seed(1))

    
    print('TEST START!')
    
    # test(model, device, test_data_laoder)
    test(model, test_data_laoder)

    print('FINISH!!!!!')



    
