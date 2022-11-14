import time
import random
import math
import jamotools
import logging

import torch
from torchmetrics.functional import char_error_rate
from transformers import AdamW
from transformers import get_scheduler

from model.vpmodel import SpeechTransformer, LabelSmoothingLoss
from dataloader import *
from Label2id import *

import warnings
import gc

warnings.filterwarnings(action='ignore')
gc.collect()

print('torch version: ',torch.__version__)

# for reproducibility
seed = 1051

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def make_directory(directory: str) -> str:
    """경로가 없으면 생성
    Args:
        directory (str): 새로 만들 경로

    Returns:
        str: 상태 메시지
    """

    try:
        if not os.path.isdir(directory):
            os.makedirs(directory)
            msg = f"Create directory {directory}"

        else:
            msg = f"{directory} already exists"

    except OSError as e:
        msg = f"Fail to create directory {directory} {e}"

    return msg

def get_logger(name: str, file_path: str, stream=False) -> logging.RootLogger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(file_path)

    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    if stream:
        logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger

PAD_token = 0
def mask2len(mask):
    non_mask = ~mask
    mask_len = torch.sum(non_mask, dim=-1)
    return mask_len

def target_to_sentence(sample:list, id2txt):
    target = str() 
    for id in sample:
        try:
            target += (str(id2txt[id]))
        except KeyError:
            continue

    return jamotools.join_jamos(target)

def train(model, ctc_lo, train_loader, criterion, criterion_ctc, optimizer, device, train_begin, epoch, lr_scheduler):
    print_batch = 100

    total_loss = 0.
    total_loss_att = 0.
    total_loss_ctc = 0.

    total_num = 0
    batch = 0

    epoch_totcer = []

    model.train()

    print('train() start')

    total_batch_size = len(train_loader)

    begin = epoch_begin = time.time()

    total_loss = 0.
    total_loss_att = 0.
    total_loss_ctc = 0.
    total_num = 0

    for vid, aud, tgt, tgt_len, mask in train_loader:
        optimizer.zero_grad()
    
        t_a = aud.shape[1]
        t_v = vid.shape[2]

        if t_a != t_v:
            min_len = min(t_a, t_v)
            vid = vid[:, :, :min_len, :, :]  
            aud = aud[:, :min_len, :]
        
        vid_len = vid.shape[0]
        vid = vid.to(device)
        aud = aud.to(device)
        tgt = tgt.to(device)
        tgt_len = tgt_len.to(device)

        target = tgt[:, 1:]

        logit, memory, memory_key_padding_mask = model(vid, vid_len, aud, tgt[:, :-1], tgt_len - 1, mask)

        y_hat = logit.max(-1)[1]

        target_ = target.contiguous().view(-1)

        # padding 제외한 value index 추출
        real_value_index = [target_ != PAD_token]

        enc_out = ctc_lo(F.dropout(memory, p=0.1))
        enc_out = enc_out.transpose(0, 1)
        encoder_log_probs =enc_out.log_softmax(2)

        targets = target
        output_lengths = mask2len(memory_key_padding_mask)
        target_lengths = tgt_len - 1
        
        loss_ctc = criterion_ctc(encoder_log_probs, targets, output_lengths, target_lengths) / targets.size(0)
        
        loss_att, loss_value = criterion(logit.contiguous(), target.contiguous())

        if torch.isfinite(loss_att) and torch.isfinite(loss_ctc):
            total_loss_att += loss_att.item()
            total_loss_ctc += loss_ctc.item()

            loss = (0.7 * loss_att) + (0.3 * loss_ctc)

            total_loss += loss.item()
            total_num += 1 # target_[real_value_index].size(0)
        else:
            loss = loss_att
                
        predictions, references = [], []
        for sample, label in zip(y_hat, tgt):
            prediction = target_to_sentence(sample.tolist(), id2txt)
            reference = target_to_sentence(label.tolist(), id2txt)
            predictions.append(prediction)
            references.append(reference)
        cer = char_error_rate(predictions=predictions, references=references)
        # print(f'char_error_rate:{cer}')

        epoch_totcer.append(cer)

        loss.backward()
        # optimizer.step_and_update_lr()

        # compute the gradient norm to check if it is normal or not
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0) # max_norm=10.0
        # logger.info(f'grad norm={grad_norm}')
        
        if math.isnan(grad_norm):
            print('grad norm is nan. Do not update model.')
        else:
            optimizer.step()

        # optimizer.step()
        lr_scheduler.step()

        if batch % print_batch == 0:
            current = time.time()
            elapsed = current - begin
            epoch_elapsed = (current - epoch_begin) / 60.0
            train_elapsed = (current - train_begin) / 3600.0

            avg_cer = np.mean(epoch_totcer)

            print('epoch: {:4d}, batch: {:5d}/{:5d}, lr: {:.16f},\nloss: {:.8f}, loss_att: {:.8f}, loss_ctc: {:.8f}, cer: {:.8f}, elapsed: {:6.2f}s {:6.2f}m {:6.2f}h'.format(
                    epoch, batch,
                    total_batch_size,
                    # scheduler.get_lr()[0],
                    # optimizer.get_lr(),
                    optimizer.param_groups[0]['lr'],
                    total_loss / total_num,
                    total_loss_att / total_num,
                    total_loss_ctc / total_num,
                    avg_cer,
                    elapsed, epoch_elapsed, train_elapsed))
            begin = time.time()

        batch += 1

    print('train() completed')
    train_cer = np.mean(epoch_totcer)
    return total_loss / total_num, train_cer
        

def evaluate(model, ctc_lo, valid_loader, criterion, criterion_ctc, device):
    total_loss = 0.
    total_num = 0

    epoch_totcer = []

    model.eval()

    print('evaluate() start')

    with torch.no_grad():
        for vid, aud, tgt, tgt_len, mask in valid_loader:
            optimizer.zero_grad()
        
            t_a = aud.shape[1]
            t_v = vid.shape[2]

            if t_a != t_v:
                min_len = min(t_a, t_v)
                vid = vid[:, :, :min_len, :, :]  
                aud = aud[:, :min_len, :]
            
            vid_len = vid.shape[0]
            vid = vid.to(device)
            aud = aud.to(device)
            tgt = tgt.to(device)
            tgt_len = tgt_len.to(device)

            target = tgt[:, 1:]

            logit, memory, memory_key_padding_mask = model(vid, vid_len, aud, tgt[:, :-1], tgt_len - 1, mask)
            
            y_hat = logit.max(-1)[1]

            target_ = target.contiguous().view(-1)

            # padding 제외한 value index 추출
            real_value_index = [target_ != PAD_token]

            enc_out = ctc_lo(F.dropout(memory, p=0.1))
            enc_out = enc_out.transpose(0, 1)
            encoder_log_probs =enc_out.log_softmax(2)

            targets = target
            output_lengths = mask2len(memory_key_padding_mask)
            target_lengths = tgt_len - 1
            
            loss_ctc = criterion_ctc(encoder_log_probs, targets, output_lengths, target_lengths) / targets.size(0)
            
            loss_att, loss_value = criterion(logit.contiguous(), target.contiguous())

            if torch.isfinite(loss_att) and torch.isfinite(loss_ctc):
                
                loss = (0.7 * loss_att) + (0.3 * loss_ctc)
                
                total_loss += loss.item()
                total_num += 1 # target_[real_value_index].size(0)
            else:
                loss = loss_att
            
            # print(f'loss:{loss}')
            
            predictions, references = [], []
            for sample, label in zip(y_hat, tgt):
                prediction = target_to_sentence(sample.tolist(), id2txt)
                reference = target_to_sentence(label.tolist(), id2txt)
                predictions.append(prediction)
                references.append(reference)
            cer = char_error_rate(predictions=predictions, references=references)
            # print(f'char_error_rate:{cer}')

            epoch_totcer.append(cer)

    print('evaluate() completed')
    valid_cer = np.mean(epoch_totcer)
    return total_loss / total_num, valid_cer


def load(filename, model, optimizer, logger):
    # state = torch.load(filename)
    state = torch.load(filename, map_location=torch.device('cpu'))
    model.load_state_dict(state['model'])
    if 'optimizer' in state and optimizer:
        optimizer.load_state_dict(state['optimizer'])
    logger.info('Model loaded : {}'.format(filename))


def save(filename, model, optimizer, logger):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, filename)
    logger.info('Model saved')



if __name__ == '__main__':
    
    max_length = 256 
    batch_size = 16
    epochs = 50
    learning_rate = 5e-5 # 0.0015 # (espnet)
    device = torch.device("cuda:0")
    max_vocab_size = 71
    model = SpeechTransformer(vocab_size=max_vocab_size, device=device)
    model = model.to(device)
    # print(model)

    count_parameters(model)


    dset_args = {'video_path' : '/home/ubuntu/nia/Final_Test/data/Video_npy',
                    'audio_path' : '/home/ubuntu/nia/Final_Test/data/Noisy_wav',
                    'transcript_path' : '/home/ubuntu/nia/Final_Test/data/id_labels',
                    'batch_size' : 8,
                    'workers' : 8
                }

    vid_args = VideoArgs()
    vid_args.batch_size = 8
    dset_loaders = get_data_loaders(dset_args)
    train_loader = dset_loaders['train']
    valid_loader = dset_loaders['val']

    txt2id, id2txt = vocab_generator()

    blank_id = txt2id['<blank>']
    criterion_ctc = torch.nn.CTCLoss(blank=blank_id, reduction='sum', zero_infinity=True)
    ctc_lo = torch.nn.Linear(256, len(txt2id)).to(device)
    criterion = LabelSmoothingLoss(size=len(txt2id), padding_idx=0, smoothing=0.1)
    optimizer = AdamW(model.parameters(),
                  lr=learning_rate, # 0.01,
                  eps=1e-08,
                  weight_decay=0.01,
                  correct_bias=True)

    train_start_time = train_begin = time.time()

    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=8000,
        num_training_steps=num_training_steps
    )

    logger = get_logger(name='train',
                file_path=os.path.join('.', 'train_log.log'), stream=True)

    for epoch in range(0, epochs):
        epoch_start_time = time.time()

        # train function
        train_loss, train_cer = train(model, ctc_lo, train_loader, criterion, criterion_ctc, optimizer, device, train_begin, epoch, lr_scheduler)
        logger.info('Epoch %d (Training) Loss %0.8f CER %0.8f' % (epoch, train_loss, train_cer))

        # evaluate function
        valid_loss, valid_cer = evaluate(model, ctc_lo, valid_loader, criterion, criterion_ctc, device)
        logger.info('Epoch %d (Evaluate) Loss %0.8f CER %0.8f' % (epoch, valid_loss, valid_cer))
        
        # make_directory('checkpoint')
        save(os.path.join('checkpoint', f"model_{epoch:03d}.pt"), model=model, optimizer=optimizer, logger=logger)

        epoch_end_time = time.time()
