import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import math
import numpy as np
import pdb

import os
import glob
import torch
import random
import sys
import torch.optim as optim

class VideoArgs:
    def __init__(self,
                 num_classes:int = 1024, #500
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
                 label_path='./home/ubuntu/nia/Final_Test/data/id_labels',
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
        self.batch_size = 16
        self.config_path = None
        self.video_dir = video_dir
        self.epochs = 80 
        self.extract_feats = False,
        self.init_epoch = 0,
        self.label_path = label_path
        self.logging_dir = logging_dir
        self.lr = 0.0003
        self.modality = 'video'
        self.model_path = None
        self.mouth_embedding_out_path = None
        self.mouth_patch_path = None
        self.optimizer = 'adamw'
        self.test = False
        self.training_mode = 'tcn'
        self.workers = 8

def threeD_to_2D_tensor(x):
    n_batch, n_channels, s_time, sx, sy = x.shape
    x = x.transpose(1, 2)
    return x.reshape(n_batch*s_time, n_channels, sx, sy)


def _average_batch(x, lengths, B):
    return torch.stack( [torch.mean( x[index][:,0:i], 1 ) for index, i in enumerate(lengths)],0 )

class MultiscaleMultibranchTCN(nn.Module):
    def __init__(self, input_size, num_channels, num_classes, vid_args, dropout, relu_type, dwpw=True):
        super(MultiscaleMultibranchTCN, self).__init__()
        self.kernel_sizes = vid_args.kernel_size
        self.num_kernels = len( self.kernel_sizes )
        self.mb_ms_tcn = MultibranchTemporalConvNet(input_size, num_channels, vid_args, dropout=dropout, relu_type=relu_type, dwpw=dwpw)
        self.tcn_output = nn.Linear(num_channels[-1], 120)

        self.consensus_func = _average_batch

    # def forward(self, x, lengths, B):
    #     # x needs to have dimension (N, C, L) in order to be passed into CNN
    #     xtrans = x.transpose(1, 2)
    #     out = self.mb_ms_tcn(xtrans)
    #     out = self.consensus_func( out, lengths, B )
    #     return self.tcn_output(out)
    def forward(self, x, lengths, B):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        # print(f'x:{x.size()}')
        xtrans = x.transpose(1, 2)
        out = self.mb_ms_tcn(xtrans)
        out = out.transpose(1, 2)
        out = self.tcn_output(out)
        # out = self.consensus_func( out, lengths, B )
        return out

class Lipreading(nn.Module):
    def __init__( self, modality='video', hidden_dim:int=256, backbone_type='shufflenet', num_classes:int=1024,
                  relu_type='prelu', width_mult=1.0, extract_feats=False):
        super(Lipreading, self).__init__()
        self.extract_feats = extract_feats
        self.backbone_type = backbone_type
        self.modality = modality
        
        vid_args=VideoArgs()

        if self.backbone_type == 'shufflenet':
            assert width_mult in [0.5, 1.0, 1.5, 2.0], "Width multiplier not correct"
            shufflenet = ShuffleNetV2( input_size=96, width_mult=width_mult)
            self.trunk = nn.Sequential( shufflenet.features, shufflenet.conv_last, shufflenet.globalpool)
            self.frontend_nout = 24
            self.backend_out = 1024 if width_mult != 2.0 else 2048
            self.stage_out_channels = shufflenet.stage_out_channels[-1]
        elif self.backbone_type == 'resnet':
            self.frontend_nout = 64
            self.backend_out = 512
            self.trunk = ResNet(BasicBlock, [2, 2, 2, 2], relu_type=relu_type)
            
        frontend_relu = nn.PReLU(num_parameters=self.frontend_nout) if relu_type == 'prelu' else nn.ReLU()
        self.frontend3D = nn.Sequential(
                    nn.Conv3d(1, self.frontend_nout, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
                    nn.BatchNorm3d(self.frontend_nout),
                    frontend_relu,
                    nn.MaxPool3d( kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)))
        
        # 그냥 멀티브랜치 하고싶다고 해....
        tcn_class = MultiscaleMultibranchTCN
        self.tcn = tcn_class( input_size=self.backend_out,
                              num_channels=[hidden_dim*len(vid_args.kernel_size)*int(vid_args.width_mult)]*vid_args.num_layers,
                              num_classes=num_classes,
                              vid_args=vid_args,
                              dropout=vid_args.dropout,
                              relu_type=relu_type,
                              dwpw=vid_args.dwpw,
                            )
        # -- initialize
        self._initialize_weights_randomly()


    def forward(self, x, lengths):
        if self.modality == 'video':
            B, C, T, H, W = x.size()
            x = self.frontend3D(x)
            Tnew = x.shape[2]    # outpu should be B x C2 x Tnew x H x W
            x = threeD_to_2D_tensor( x )
            x = self.trunk(x)
            if self.backbone_type == 'shufflenet':
                x = x.view(-1, self.stage_out_channels)
            x = x.view(B, Tnew, x.size(1))
        elif self.modality == 'raw_audio':
            B, C, T = x.size()
            x = self.trunk(x)
            x = x.transpose(1, 2)
            lengths = [_//640 for _ in lengths]

        return x if self.extract_feats else self.tcn(x, lengths, B)


    def _initialize_weights_randomly(self):

        use_sqrt = True

        if use_sqrt:
            def f(n):
                return math.sqrt( 2.0/float(n) )
        else:
            def f(n):
                return 2.0/float(n)

        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                n = np.prod( m.kernel_size ) * m.out_channels
                m.weight.data.normal_(0, f(n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.Linear):
                n = float(m.weight.data[0].nelement())
                m.weight.data = m.weight.data.normal_(0, f(n))

class Chomp1d(nn.Module):
    def __init__(self, chomp_size, symm_chomp):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
        self.symm_chomp = symm_chomp
        if self.symm_chomp:
            assert self.chomp_size % 2 == 0, "If symmetric chomp, chomp size needs to be even"
    def forward(self, x):
        if self.chomp_size == 0:
            return x
        if self.symm_chomp:
            return x[:, :, self.chomp_size//2:-self.chomp_size//2].contiguous()
        else:
            return x[:, :, :-self.chomp_size].contiguous()
        

class ConvBatchChompRelu(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, relu_type, dwpw=False):
        super(ConvBatchChompRelu, self).__init__()
        self.dwpw = dwpw
        if dwpw:
            self.conv = nn.Sequential(
                # -- dw
                nn.Conv1d( n_inputs, n_inputs, kernel_size, stride=stride,
                           padding=padding, dilation=dilation, groups=n_inputs, bias=False),
                nn.BatchNorm1d(n_inputs),
                Chomp1d(padding, True),
                nn.PReLU(num_parameters=n_inputs) if relu_type == 'prelu' else nn.ReLU(inplace=True),
                # -- pw
                nn.Conv1d( n_inputs, n_outputs, 1, 1, 0, bias=False),
                nn.BatchNorm1d(n_outputs),
                nn.PReLU(num_parameters=n_outputs) if relu_type == 'prelu' else nn.ReLU(inplace=True)
            )
        else:
            self.conv = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                               stride=stride, padding=padding, dilation=dilation)
            self.batchnorm = nn.BatchNorm1d(n_outputs)
            self.chomp = Chomp1d(padding,True)
            self.non_lin = nn.PReLU(num_parameters=n_outputs) if relu_type == 'prelu' else nn.ReLU()

    def forward(self, x):
        if self.dwpw:
            return self.conv(x)
        else:
            out = self.conv( x )
            out = self.batchnorm( out )
            out = self.chomp( out )
            return self.non_lin( out )

class MultibranchTemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_sizes, stride, dilation, padding, dropout=0.2, 
                 relu_type = 'relu', dwpw=False):
        super(MultibranchTemporalBlock, self).__init__()
        
        self.kernel_sizes = kernel_sizes
        self.num_kernels = len( kernel_sizes )
        self.n_outputs_branch = n_outputs // self.num_kernels
        assert n_outputs % self.num_kernels == 0, "Number of output channels needs to be divisible by number of kernels"



        for k_idx,k in enumerate( self.kernel_sizes ):
            cbcr = ConvBatchChompRelu( n_inputs, self.n_outputs_branch, k, stride, dilation, padding[k_idx], relu_type, dwpw)
            setattr( self,'cbcr0_{}'.format(k_idx), cbcr )
        self.dropout0 = nn.Dropout(dropout)
        
        for k_idx,k in enumerate( self.kernel_sizes ):
            cbcr = ConvBatchChompRelu( n_outputs, self.n_outputs_branch, k, stride, dilation, padding[k_idx], relu_type, dwpw)
            setattr( self,'cbcr1_{}'.format(k_idx), cbcr )
        self.dropout1 = nn.Dropout(dropout)

        # downsample?
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if (n_inputs//self.num_kernels) != n_outputs else None
        
        # final relu
        if relu_type == 'relu':
            self.relu_final = nn.ReLU()
        elif relu_type == 'prelu':
            self.relu_final = nn.PReLU(num_parameters=n_outputs)

    def forward(self, x):

        # first multi-branch set of convolutions
        outputs = []
        for k_idx in range( self.num_kernels ):
            branch_convs = getattr(self,'cbcr0_{}'.format(k_idx))
            outputs.append( branch_convs(x) )
        out0 = torch.cat(outputs, 1)
        out0 = self.dropout0( out0 )

        # second multi-branch set of convolutions
        outputs = []
        for k_idx in range( self.num_kernels ):
            branch_convs = getattr(self,'cbcr1_{}'.format(k_idx))
            outputs.append( branch_convs(out0) )
        out1 = torch.cat(outputs, 1)
        out1 = self.dropout1( out1 )
                
        # downsample?
        res = x if self.downsample is None else self.downsample(x)

        return self.relu_final(out1 + res)

class MultibranchTemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, vid_args, dropout=0.2, relu_type='relu', dwpw=True):
        super(MultibranchTemporalConvNet, self).__init__()

        self.ksizes = vid_args.kernel_size
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]

            padding = [ (s-1)*dilation_size for s in self.ksizes]            
            layers.append( MultibranchTemporalBlock( in_channels, out_channels, self.ksizes, 
                stride=1, dilation=dilation_size, padding = padding, dropout=dropout, relu_type = relu_type,
                dwpw=dwpw) )

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)        

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups
    
    # reshape
    x = x.view(batchsize, groups, 
        channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, benchmodel):
        super(InvertedResidual, self).__init__()
        self.benchmodel = benchmodel
        self.stride = stride
        assert stride in [1, 2]

        oup_inc = oup//2
        
        if self.benchmodel == 1:
            #assert inp == oup_inc
            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )                
        else:                  
            self.banch1 = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )        
    
            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )
          
    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)        

    def forward(self, x):
        if 1==self.benchmodel:
            x1 = x[:, :(x.shape[1]//2), :, :]
            x2 = x[:, (x.shape[1]//2):, :, :]
            out = self._concat(x1, self.banch2(x2))
        elif 2==self.benchmodel:
            out = self._concat(self.banch1(x), self.banch2(x))

        return channel_shuffle(out, 2)

class ShuffleNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=2.):
        super(ShuffleNetV2, self).__init__()
        
        assert input_size % 32 == 0, "Input size needs to be divisible by 32"
        
        self.stage_repeats = [4, 8, 4]
        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if width_mult == 0.5:
            self.stage_out_channels = [-1, 24,  48,  96, 192, 1024]
        elif width_mult == 1.0:
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif width_mult == 1.5:
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            self.stage_out_channels = [-1, 24, 244, 488, 976, 2048]
        else:
            raise ValueError(
                """Width multiplier should be in [0.5, 1.0, 1.5, 2.0]. Current value: {}""".format(width_mult))

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.conv1 = conv_bn(3, input_channel, 2)    
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.features = []
        # building inverted residual blocks
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage+2]
            for i in range(numrepeat):
                if i == 0:
                #inp, oup, stride, benchmodel):
                    self.features.append(InvertedResidual(input_channel, output_channel, 2, 2))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, 1))
                input_channel = output_channel
                
                
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building last several layers
        self.conv_last  = conv_1x1_bn(input_channel, self.stage_out_channels[-1])
        self.globalpool = nn.Sequential(nn.AvgPool2d(int(input_size/32)))              
        
        # building classifier
        self.classifier = nn.Sequential(nn.Linear(self.stage_out_channels[-1], n_class))

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.features(x)
        x = self.conv_last(x)
        x = self.globalpool(x)
        x = x.view(-1, self.stage_out_channels[-1])
        x = self.classifier(x)
        return x



if __name__=='__main__':
    vid_args = VideoArgs()
    model = Lipreading( modality=vid_args.modality,
                    num_classes=vid_args.num_classes,
                    vid_args=vid_args,
                    backbone_type=vid_args.backbone_type,
                    relu_type=vid_args.relu_type,
                    width_mult=vid_args.width_mult,
                    extract_feats=vid_args.extract_feats)
    # B, C, T, H, W
    # shuffle net 가로 세로는 무조건 96
    tmp = torch.randn((4, 1, 15, 96, 96))
    model(tmp, tmp.shape[0]).shape
