from torch import nn
from torch.nn import Sequential
import torch
from hw_asr.base import BaseModel
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class DeepSpeech2(BaseModel):
    def __init__(self, n_feats, n_class, fc_hidden=512, **batch):
        super(DeepSpeech2, self).__init__(n_feats, n_class, **batch)
        self.conv_part = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(21, 6)),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(11, 6)),
            nn.BatchNorm2d(32),
            nn.Tanh()
        )

        input_size_ = self.transform_input_lengths(n_feats, dim=0) * 32
        self.rnn = nn.LSTM(input_size=input_size_, hidden_size=fc_hidden, num_layers=4, bidirectional=True, batch_first=True)

        self.fc = nn.Sequential(
                nn.Linear(fc_hidden * 2, fc_hidden),
                nn.ReLU(),
                nn.Linear(fc_hidden, n_class)
        )

    def forward(self, spectrogram, **batch):
        x = self.conv_part(spectrogram.unsqueeze(1)) #add channel by unsqueeze
        # x = (batch, channels, H, Time)
        # we do (batch, channels * H, Time) -> (batch, Time, channels * H)
        x = x.view(x.shape[0], -1, x.shape[-1]).transpose(1,2)
        
        seq_lens = self.transform_input_lengths(batch['spectrogram_length'], dim=1)
        x = pack_padded_sequence(x, lengths=seq_lens, batch_first=True, enforce_sorted=False) #unpad x to feed rnn
        x = self.rnn(x)[0]
        x = pad_packed_sequence(x, batch_first=True)[0] #pad x
        return {'logits' : self.fc(x)}

    def transform_input_lengths(self, input_lengths, dim=1):
        #we change len only by convs 
        kernel_size1 = (41, 11)
        stride1 = (2, 2)
        paddings1 = (21, 6)

        kernel_size2 = (21, 11)
        stride2 = (2, 1)
        paddings2 = (11, 6)


        past_conv1_len = (input_lengths + 2 * paddings1[dim] - (kernel_size1[dim] - 1) - 1) // stride1[dim] + 1
        past_conv2_len = (past_conv1_len + 2 * paddings2[dim] - (kernel_size2[dim] - 1) - 1) // stride2[dim] + 1
        return past_conv2_len
