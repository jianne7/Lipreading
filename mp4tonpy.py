import os
import cv2
import glob
import numpy as np
import argparse
import pdb


def extract_opencv(filename):
    video = []
    cap = cv2.VideoCapture(filename)
    while(cap.isOpened()):
        ret, frame = cap.read() # BGR
        if ret:
            video.append(frame)
        else:
            break
    cap.release()
    video = np.array(video)
    return video[...,::-1]

if __name__=='__main__':
    noisy = sorted(glob.glob('data/Noisy/*_sr.wav'), key=lambda x: x.split('_')[-2])
    video = ['data/Video25/' + x.split('_')[-2] + '.mp4' for x in noisy]

    data_folder = './data/Video25'
    filenames = video
    i=1
    for idx, filename in enumerate(filenames):
        if i > 63239:
            data = extract_opencv(filename) 
            path_to_save = os.path.join(data_folder.replace('Video25','Video_npy'),
                                        f'{idx}_' + filename.split('/')[-1][:-4]+'.npy')
        # if not os.path.exists(os.path.dirname(path_to_save)):
        #     print("path_to_save doesn't exist@")
        #     try:
        #         os.makedirs(os.path.dirname('/data/Video_npy'))
        #     except:
        #         pass
            np.save(path_to_save, data)
            i+=1
        else:
            i+=1