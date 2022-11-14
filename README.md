# VP Model 설명
## 필요 라이브러리  
numpy==1.20.0  
cudatoolkit==11.3.1  
ffmpeg==1.4  
jamotools==0.1.10  
librosa==0.8.1  
numba==0.54.1  
numpy==1.20.3  
opencv-python==4.5.4.60  
pandas==1.3.5  
pesq==0.0.3  
pydub==0.25.1  
python==3.8.12  
soundfile==0.10.3.post1  
torch==1.9.0+cu111  
torchaudio==0.9.0  
torchmetrics==0.6.2  
torchvision==0.10.0+cu111  
tqdm==4.62.3  
transformers==4.13.0  
accelerate==0.5.1
  
  
## 작업 폴더 구성 및 설명  
 '''
- 작업 폴더 설명
    
    Final_Test
      ├── data : (pre processed) data
      │   ├── grapheme_labels : 레이블을 자소로 변환한 파일 폴더
      │   ├── id_labels : 자소 레이블을 id 레이블로 변환한 파일 폴더
      │   ├── Noise : Noise 폴더
      │   ├── Noisy : Train 음성 파일에 Noise를 섞은 파일 폴더
      │   ├── Noisy_wav : Noise 제거 모델로 추론하여 노이즈를 제거한 음성 파일 폴더
      │   ├── raw_data_zip : Train.zip, Test.zip, Noise.zip 등을 모아둔 폴더
      │   ├── Test : 대회에서 제공한 Test 파일 폴더
      │   │   ├── Audio : 원본 Audio 파일
      │   │   ├── Video : 원본 Video 파일
      │   │   ├── Denoise_wav : 노이즈 처리 모델로 전처리한 Audio 파일
      │   │   ├── Video_fps : 비디오 프레임 전처리한 폴더
      │   │   ├── Video_npy : 비디오를 넘파이로 변환한 폴더
      │   ├── Train : 대회에서 제공한 Train 파일 폴더
      │   ├── Video_npy : 25프레임 비디오를 넘파이로 변환한 파일 폴더
      │   ├── Video25 : 30프레임 원본 mp4 파일을 25프레임으로 변환한 파일 폴더
      │   ├── 기타
      ├── model : model directory
      │   ├── pretrained_weights : 노이즈 제거 모델 사전학습 가중치
      │   │   ├── fullsubnet_best_model_58epochs.tar : public하게 공개되어 있는 사전학습 가중치
      │   ├── FusionNet.py : 비디오넷과 오디오넷을 결합하는 모델 아키텍쳐
      │   ├── Videonet.py : 비디오 파일 처리 모델
      │   ├── vpmodel.py : 음성 파일 처리 및 최종 통합 모델 아키텍쳐
      ├── checkpoint : model save / load 관련
      │   ├── epoch별 저장한 모델
      ├── dataloader.py : 데이터셋 및 데이터 로더 구현
      ├── fsn.py : 노이즈와 합성한 wav 파일을 노이즈 제거 (노이즈 제거 모델)
      ├── Label2id.py : 자소 변환 및 txt2id 변환
      ├── mp4tonpy.py : 25프레임 비디오를 넘파이로 변환
      ├── Noise.py : 노이즈와 wav 파일 합성  
      ├── fps_changer.py : 30프레임 원본 mp4를 25프레임으로 변환
      ├── main.py : 최종 실행
      ├── README.md : 설명
 '''


## 진행 절차 설명  
1. 대회에서 제공한 Train 폴더의 음성 데이터 전처리를 실시한다  
    1-1. Noise.py를 사용해 음성데이터에 노이즈를 섞는다  
            * 경로
            # song_list : 원본 Audio 경로
            # noise_list : 원본 Noise 경로
            # output_path : 새롭게 생성할 Audio + Noise 파일 폴더 경로
            song_list = '/home/ubuntu/nia/Final_Test/data/Train/Train/Audio'
            noise_list = '/home/ubuntu/nia/Final_Test/data/Noise/'
            output_path = '/home/ubuntu/nia/Final_Test/data/Noisy'
    1-2. 시드를 고정하고 Noise 제거 모델을 활용하여 wav를 추출한다 (fsn.py)
            * 경로
            # 훈련 시 fsn.py에 있는 주석을 **꼭** 해제해주세요. (주석은 추론돌릴 때만 있으면 됩니다)
            # checkpoint_path : FullSubNet Pretrained Weights 경로
            # output_dir : 딥러닝 모델을 사용하여 wav에 존재하는 Noise 제거
            # dataset_dir_list : 1-1에서 생성한 Noisy 파일 경로
            checkpoint_path = '/home/ubuntu/nia/Final_Test/model/pretrained_weights/fullsubnet_best_model_58epochs.tar'
            output_dir = '/home/ubuntu/nia/Final_Test/data/Noisy_wav'
            dataset_dir_list = ["/home/ubuntu/nia/Final_Test/data/Noisy",] 
2. 대회에서 제공한 Train 폴더의 영상 데이터 전처리를 실시한다  
    2-1. fps_changer.py를 사용하여 30프레임 mp4 파일을 25프레임을 변환하여 저장한다 
            * 경로
            # data_folder : 원본 Video 경로
            # fps_folder : 25 프레임으로 변경(전처리)한 파일 경로
                fps_changer = FpsChanger(
                                data_folder="/home/ubuntu/nia/Final_Test/data/Train/Train/Video",
                                fps_folder="/home/ubuntu/nia/Final_Test/data/Train/Train/Video25",
                                target_fps=25,
                            )
    2-2. 25 프레임 비디오를 넘파이로 변환하여 저장한다 (mp4tonpy.py)  
            * 경로
            # Video_npy 폴더를 미리 **꼭** 생성할 것 (자세한 경로는 위에 디렉토리 설명 참조)
3. Label2id.py를 실행하여 레이블을 자모 형태로 변환한다
    3-1. 
            * 경로
            # data 폴더 안에 grapheme_labels, id_labels를 반드시 생성할 것
            
            # data 안에 label 폴더가 있을 시 data 폴더 경로를 넣을 것
            jamo('/home/ubuntu/nia/Final_Test/data', label)

            # audio_paths : Noisy_wav 폴더 경로
            # video_paths : Video_npy 폴더 경로
            # grapheme_labels : 자모 처리가 끝난 폴더 경로
            audio_paths = sorted(glob.glob('/home/ubuntu/nia/Final_Test/data/Noisy_wav/*_sr.wav'), key=lambda x: x.split('_')[-2])
            video_paths = sorted(glob.glob('/home/ubuntu/nia/Final_Test/data/Video_npy/*.npy'), key=lambda x: x.split('_')[-1])
            label_paths = sorted(glob.glob('/home/ubuntu/nia/Final_Test/data/grapheme_labels/*.txt'), key=lambda x: x.split('_')[-1])
4. main.py을 실행한다 (1 gpu)  
    $ python main.py
    4-1. 비디오넷, 오디오넷으로 구성된 멀티모달이 퓨전넷을 통과하여 결합된다
    4-2. 최종적으로 Decoder에 넣어 예측값을 구한다

4.5 멀티 gpu로 훈련할 시 main_mg.py를 실행한다 (강추)
    $ accelerate launch main_mg.py

5. Inference의 경우 Inference.py를 실행하면 된다
    여기는 Test 폴더 안에 미리 경로를 생성해둘 필요 없다 (폴더 생성 코드 존재)
    # video_path : 비디오 원본 폴더
    # video25_path : 비디오 프레임 전처리 후 폴더
    # video_npy_path : 비디오 넘파이 파일 폴더
    # audio_path : 오디오 원본 폴더
    # fsn_checkpoint_path : FSN Pretrained Weights 저장 경로
    # output_dir : FSN 통과 후 경로
    
    video_path = "/home/ubuntu/nia/Final_Test/data/Test/Video"
    video25_path = "/home/ubuntu/nia/Final_Test/data/Test/Video_fps"
    video_npy_path = "/home/ubuntu/nia/Final_Test/data/Test/Video_npy"
    audio_path = "/home/ubuntu/nia/Final_Test/data/Test/Audio"
    fsn_checkpoint_path = '/home/ubuntu/nia/Final_Test/model/pretrained_weights/fullsubnet_best_model_58epochs.tar'
    output_dir = '/home/ubuntu/nia/Final_Test/data/Test/Denoise_wav'

## Pretrained Weights 관련 참조 주소
github.com/haoxiangsnr/FullSubNet
**public 하게 공개되어 있는 사전학습 가중치**