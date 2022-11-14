import glob
import os


def sentence_to_target(sentence, txt2id):
    target = str()
    for ch in sentence:
        try:
            target += (str(txt2id[ch]) + ' ')
        except KeyError:
            continue
    return target[:-1]


def label_loader(files:list):
    labels = []
    for idx, file in enumerate(files):
        with open(file, 'r') as f:
            data = f.read()
        labels.append(data)
    return labels

def vocab_generator():
    txt2id = {f'{g}':i+5 for i, g in enumerate('ㄱ ㄲ ㄴ ㄷ ㄸ ㄹ ㅁ ㅂ ㅃ ㅅ ㅆ ㅇ ㅈ ㅉ ㅊ ㅋ ㅌ ㅍ ㅎ ㅏ ㅐ ㅑ ㅒ ㅓ ㅔ ㅕ ㅖ ㅗ ㅘ ㅙ ㅚ ㅛ ㅜ ㅝ ㅞ ㅟ ㅠ ㅡ ㅢ ㅣ ㄳ ㄵ ㄶ ㄺ ㄻ ㄼ ㄽ ㄾ ㄿ ㅀ ㅄ 1 2 3 4 5 6 7 8 9 0 a x A X'.split(' '))}
    txt2id['<pad>'] = 0
    txt2id['<unk>'] = 1
    txt2id['<sos>'] = 2
    txt2id['<eos>'] = 3
    txt2id[' '] = 4
    txt2id['<blank>'] = 70
    id2txt = {int(value):key for key, value in txt2id.items()}

    return txt2id, id2txt


if __name__=='__main__':

    txt2id, id2txt = vocab_generator()

    audio_paths = sorted(glob.glob('/home/ubuntu/nia/Final_Test/data/Noisy_wav/*_sr.wav'), key=lambda x: x.split('_')[-2])
    video_paths = sorted(glob.glob('/home/ubuntu/nia/Final_Test/data/Video_npy/*.npy'), key=lambda x: x.split('_')[-1])
    label_paths = sorted(glob.glob('/home/ubuntu/nia/Final_Test/data/grapheme_labels/*.txt'), key=lambda x: x.split('_')[-1])
    labels = label_loader(label_paths)

    for label, filename in zip(labels, label_paths):
        char_id_label = sentence_to_target(label, txt2id)
        with open(f'/home/ubuntu/nia/Final_Test/data/id_labels/{filename.split("/")[-1]}', 'w') as f:
            f.write(char_id_label)

