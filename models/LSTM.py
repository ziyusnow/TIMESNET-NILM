import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.hidden_size = 128
        self.num_layers = 2
        self.enc_in=configs.enc_in
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(p=0.3)
        self.lstm1=nn.LSTM(self.enc_in, self.hidden_size, self.num_layers, batch_first=True)
        self.lstm2=nn.LSTM(self.enc_in, self.hidden_size, self.num_layers, batch_first=True)        
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 2*self.hidden_size),
            self.relu,
            nn.Linear(2*self.hidden_size, 4*self.hidden_size),
            self.relu,
            nn.Linear(4*self.hidden_size, 4),
            self.relu,
            self.dropout)
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, 2*self.hidden_size),           
            self.relu,
            self.dropout,
            nn.Linear(2*self.hidden_size, 4),
            )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # h0 = torch.zeros(self.num_layers, x_enc.size(0), self.hidden_size).to(x_enc.device)
        # c0 = torch.zeros(self.num_layers, x_enc.size(0), self.hidden_size).to(x_enc.device)

        #out, _ = self.lstm(x_enc, (h0, c0))
        out, _ = self.lstm1(x_enc)
        state, _ = self.lstm2(x_enc)
        out=self.relu(out)
        out=self.dropout(out)
        state=self.relu(state)
        state=self.dropout(state)
        state = self.classifier(state[:, -1:, :])
        out = self.fc(out[:, -1:, :])
        #out= state * out
        return out,state