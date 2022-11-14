import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import dataset
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import math
import numpy as np
import pdb
import cv2
import os
import glob
import random
import librosa
import sys
import torch.optim as optim
# from skimage import transform as tf
import torchaudio

class VideoArgs:
    def __init__(self,
                 num_classes:int = 500,
                 interval:int = 50, 
                 backbone_type:str = "shufflenet",
                 relu_type:str = "relu",
                 dropout:float = 0.2,
                 dwpw:bool = True,
                 kernel_size:list = [3, 5, 7],
                 num_layers:int = 4,
                 width_mult:float = 1.0,
                 allow_size_mismatch = False,
                 alpha = 0.4,
                 batch_size = 32,
                 config_path = None,
                 video_dir='/home/ubuntu/nia/Final_Test/data/Video_npy',
                 epochs=80, 
                 extract_feats=False,
                 init_epoch=0,
                 label_path='/home/ubuntu/nia/Final_Test/data/id_labels',
                 logging_dir='./train_logs',
                 lr = 0.0003,
                 modality='video',
                 model_path=None, # pretrained weight path
                 mouth_embedding_out_path=None,
                 mouth_patch_path=None,
                 optimizer='adamw',
                 test=False,
                 training_mode='tcn',
                 workers=8
                ):
        self.num_classes = num_classes
        self.interval = 50
        self.backbone_type = backbone_type
        self.relu_type = relu_type
        self.dropout = dropout
        self.dwpw = dwpw
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.width_mult = width_mult
        self.allow_size_mismatch = allow_size_mismatch
        self.alpha = alpha
        self.batch_size = 32
        self.config_path = None
        self.video_dir = video_dir
        self.epochs = 80 
        self.extract_feats = False,
        self.init_epoch = 0,
        self.label_path = label_path
        self.logging_dir = './train_logs'
        self.lr = 0.0003
        self.modality = 'video'
        self.model_path = None
        self.mouth_embedding_out_path = None
        self.mouth_patch_path = None
        self.optimizer = 'adamw'
        self.test = False
        self.training_mode = 'tcn'
        self.workers = 8

class FilterBank(object):
    def __init__(self, sample_rate: int = 16000, n_mels: int = 80,
                 frame_length: int = 60, frame_shift: int = 40, snip_edges=False) -> None:
        self.transforms = torchaudio.compliance.kaldi.fbank
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.frame_length = frame_length
        self.frame_shift = frame_shift

    def __call__(self, signal):
        return self.transforms(
            Tensor(signal).unsqueeze(0),
            num_mel_bins=self.n_mels,
            frame_length=self.frame_length,
            frame_shift=self.frame_shift,
            snip_edges=False
        )

class SpecAugment(object):
    def __init__(self, time_mask_num: int = 2, freq_mask_num: int = 2,
                time_mask_width_range: list = [0, 40], freq_mask_width_range: list = [0, 30]):
        self.time_mask_num = time_mask_num
        self.freq_mask_num = freq_mask_num
        self.time_mask_width_range = time_mask_width_range
        self.freq_mask_width_range = freq_mask_width_range
    def __call__(self, feature: Tensor) :
        """ Provides SpecAugmentation for audio """
        time_axis_length = feature.size(0)
        freq_axis_length = feature.size(1)
        
        # time mask
        low, high = self.time_mask_width_range
        for _ in range(self.time_mask_num):
            try:
                t = int(np.random.uniform(low=0.0, high=high))
                t0 = random.randint(0, time_axis_length - t)
                feature[t0: t0 + t, :] = 0
            except ValueError as e:
                print(f"time_mask: {e}")
            except Exception as e:
                print(f"Exception: {e}")

        # freq mask
        low, high = self.freq_mask_width_range
        for _ in range(self.freq_mask_num):
            try:
                f = int(np.random.uniform(low=low, high=high))
                f0 = random.randint(0, freq_axis_length - f)
                feature[:, f0: f0 + f] = 0
            except ValueError as e:
                print(f"freq_mask: {e}")
            except Exception as e:
                print(f"Exception: {e}")
        return feature

class Compose(object):
    """Compose several preprocess together.
    Args:
        preprocess (list of ``Preprocess`` objects): list of preprocess to compose.
    """

    def __init__(self, preprocess):
        self.preprocess = preprocess

    def __call__(self, sample):
        for t in self.preprocess:
            sample = t(sample)
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.preprocess:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RgbToGray(object):
    """Convert image to grayscale.
    Converts a numpy.ndarray (H x W x C) in the range
    [0, 255] to a numpy.ndarray of shape (H x W x C) in the range [0.0, 1.0].
    """

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Image to be converted to gray.
        Returns:
            numpy.ndarray: grey image
        """
        frames = np.stack([cv2.cvtColor(_, cv2.COLOR_RGB2GRAY) for _ in frames], axis=0)
        return frames

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Normalize(object):
    """Normalize a ndarray image with mean and standard deviation.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, frames):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        frames = (frames - self.mean) / self.std
        return frames

    def __repr__(self):
        return self.__class__.__name__+'(mean={0}, std={1})'.format(self.mean, self.std)


class CenterCrop(object):
    """Crop the given image at the center
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be cropped.
        Returns:
            numpy.ndarray: Cropped image.
        """
        t, h, w = frames.shape
        th, tw = self.size
        delta_w = int(round((w - tw))/2.)
        delta_h = int(round((h - th))/2.)
        frames = frames[:, delta_h:delta_h+th, delta_w:delta_w+tw]
        return frames


class RandomCrop(object):
    """Crop the given image at the center
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be cropped.
        Returns:
            numpy.ndarray: Cropped image.
        """
        t, h, w = frames.shape
        th, tw = self.size
        delta_w = random.randint(0, w-tw)
        delta_h = random.randint(0, h-th)
        frames = frames[:, delta_h:delta_h+th, delta_w:delta_w+tw]
        return frames

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class HorizontalFlip(object):
    """Flip image horizontally.
    """

    def __init__(self, flip_ratio):
        self.flip_ratio = flip_ratio

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be flipped with a probability flip_ratio
        Returns:
            numpy.ndarray: Cropped image.
        """
        t, h, w = frames.shape
        if random.random() < self.flip_ratio:
            for index in range(t):
                frames[index] = cv2.flip(frames[index], 1)
        return frames


class AV_Dataset(Dataset):
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
        super(AV_Dataset, self).__init__()
        
        # args : (sr, n_mels, frame_length, frame_shift)
        # 640 hop len == 16000 sr * 0.04 frame shift 
        self.transforms = FilterBank(16000, 80, 60, 40)
        self.video_paths = sorted(glob.glob(f'{video_paths}/*.npy'), key=lambda x: x.split('_')[-1]) 
        self.audio_paths = sorted(glob.glob(f'{audio_paths}/enhanced_0058/*.wav'), key=lambda x: x.split('_')[-2])
        
        self.transcripts = sorted(glob.glob(transcripts + '/*.txt'), key=lambda x: x.split('_')[-1])
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
        
        self._augment(spec_augment)
        self.shuffle()


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
            video_feature = self.parse_video(self.video_paths[index])
            audio_feature = self.parse_audio(self.audio_paths[index])
            return video_feature, audio_feature

    def parse_transcript(self, transcript):
        with open(transcript, 'r') as f:
            data = f.read()
        tokens = data.split(' ')
        transcript = list()

        transcript.append(int(self.sos_id))
        for token in tokens:
            transcript.append(int(token))
        transcript.append(int(self.eos_id))
        transcript = torch.LongTensor(transcript)
        return transcript

    def _augment(self, spec_augment):
        """ Spec Augmentation """
        if spec_augment:
            print("Applying Spec Augmentation...")
            # print(self.dataset_size)
            for idx in range(self.dataset_size):
                # spectrogram augmentation 했는지 안 했는지 유무 기록용
                self.augment_existence.append(self.SPEC_AUGMENT)
                # print(len(self.video_paths))
                self.video_paths.append(self.video_paths[idx])
                self.audio_paths.append(self.audio_paths[idx])
                self.transcripts.append(self.transcripts[idx])

    def shuffle(self):
        """ Shuffle dataset """
        # list 안에 경로 들이 각각 튜플로 저장
        tmp = list(zip(self.video_paths,self.audio_paths, self.transcripts, self.augment_existence))
        random.shuffle(tmp)
        # 튜플 안에서만 셔플
        self.video_paths,self.audio_paths, self.transcripts, self.augment_existence = zip(*tmp)

    def __len__(self):
        return len(self.audio_paths)

    def count(self):
        return len(self.audio_paths)

def _collate_fn(batch):
    
    """ functions that pad to the maximum sequence length """
    batch = sorted(batch, key=lambda sample: sample[0].shape[0], reverse=True)
    video, vid_len, audio, aud_len, label, label_len = zip(*[(video, video.shape[0],
                                                                audio, audio.shape[0], 
                                                                label, label.shape[0])
                                                               for (video, audio, label) in batch])
    # sort by sequence length for rnn.pack_padded_sequence()
    # pdb.set_trace()
    enc_mask = []
    for a, b in zip(vid_len, aud_len):
        if a > b:
            enc_mask.append(b)
        elif a < b:
            enc_mask.append(a)
        else:
            enc_mask.append(a)
    
    max_vid_sample = video[0]
    max_aud_sample = audio[0]
    max_label_sample = label[0]

    max_vid_size = max(vid_len)
    max_aud_size = max(aud_len)
    max_label_size = max(label_len)
    
    vid_feat_x = max_vid_sample.size(1) # height
    vid_feat_y = max_vid_sample.size(2) # width
#     vid_feat_c = max_vid_sample.size(3) # channel
    aud_feat_f = max_aud_sample.size(1) # frequency
    batch_size = len(audio)

    vids = torch.zeros(batch_size, max_vid_size, vid_feat_x, vid_feat_y)
    auds = torch.zeros(batch_size, max_aud_size, aud_feat_f)
    # print(auds.shape)
    targets = torch.zeros(batch_size, max_label_size).to(torch.long)
    targets.fill_(0)
    # pdb.set_trace()
    for x in range(batch_size):
        sample = batch[x]
        video_ = sample[0]
        audio_ = sample[1]
        target_ = sample[2]

        vid_len_ = video_.shape[0] # time steps
        aud_len_ = audio_.shape[0] # time steps
        vids[x, :vid_len_, :, :] = video_
        auds[x].narrow(0, 0, aud_len_).copy_(audio_)
        targets[x].narrow(0, 0, len(target_)).copy_(torch.LongTensor(target_))
    # B T H W -> B C T H W
    vids = vids.unsqueeze(1)
    # vids_leng = vids.shape[0]
    tgt_len = torch.LongTensor(label_len)
    enc_mask = torch.LongTensor(enc_mask)
    return vids, auds, targets, tgt_len, enc_mask


def get_data_loaders(args):
    # args는 딕셔너리
    dsets = {partition: AV_Dataset(
                video_paths=args['video_path'],
                audio_paths=args['audio_path'],
                transcripts=args['transcript_path'],
                spec_augment=True,
                mode=partition
                ) for partition in ['train']}
    
    
    train_dataset_size = int(0.9 * len(dsets['train']))
    valid_dataset_size = int(len(dsets['train']) - train_dataset_size)


    train_valid_sizes = [train_dataset_size, valid_dataset_size]
    train_dataset, valid_dataset = dataset.random_split(dsets['train'], train_valid_sizes)
    valid_dataset.SPEC_AUGMENT = False
    valid_dataset.mode = 'val'

    # print(len(train_dataset), len(valid_dataset)) 
    
    
    dset_loaders = {mode: DataLoader(
                        train_dataset if mode=='train' else valid_dataset,
                        batch_size=args['batch_size'],
                        shuffle=True,
                        collate_fn=_collate_fn,
                        pin_memory=False,
                        num_workers=args['workers'],
                        worker_init_fn=np.random.seed(1)) for mode in ['train', 'val']}
    return dset_loaders

# if __name__=='__main__':        
#     vid_args = VideoArgs()
    # dset_args = {'train':{'spec_augment':True,
    #             'video_path' : '/home/ubuntu/nia/Final_Test/data/Video_npy',
    #             'audio_path' : '/home/ubuntu/nia/Final_Test/data/Noisy_wav',
    #             'transcript_path' : '/home/ubuntu/nia/Final_Test/data/id_labels',
    #             'batch_size' : 16,
    #             'workers' : 1}
    #              'val':{
    #              }
    #         }
#     dset_loaders = get_data_loaders(dset_args)
#     for a, b, c, d, e in dset_loaders['train']:
#         b
#         break
