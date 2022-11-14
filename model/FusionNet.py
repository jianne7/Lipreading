import torch
from torch import nn

class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        
        input_size = 376
        hidden_size = 256
        output_size = 256
        num_layers = 4
        
        # self.device = device
      #  self.conv = torch.nn.Sequential(
      #      torch.nn.Conv2d(1, 8, [2, 10], [1,8]),
      #      torch.nn.ReLU(),
      #      torch.nn.Conv2d(8, 8, [2, 10], [1,8], padding=1),
      #      torch.nn.ReLU()
      #  )

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=0.1, bidirectional=True)     
        self.fc = nn.Linear(hidden_size * 2, output_size)
   
        
    def forward(self, aud_encoder_outputs, vid_encoder_outputs):
      #  b, t, d = vid_encoder_outputs.size()
      #  vid_encoder_outputs = vid_encoder_outputs.unsqueeze(1)
      #  vid_encoder_outputs = self.conv(vid_encoder_outputs)
      #  vid_encoder_outputs = vid_encoder_outputs.transpose(1, 2).contiguous().view(b, t, -1)

        # input: tensor of shape (batch_size, seq_length, hidden_size)
        fusion_encoder_out = torch.cat((aud_encoder_outputs,vid_encoder_outputs),-1)
        output, _ = self.lstm(fusion_encoder_out)
        output = self.fc(output)
        return output
