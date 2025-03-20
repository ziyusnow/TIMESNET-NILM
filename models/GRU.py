import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.hidden_size = 128
        self.num_layers = 2
        self.enc_in = configs.enc_in
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        
        # 将LSTM替换为GRU
        self.gru1 = nn.GRU(
            input_size=self.enc_in,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.gru2 = nn.GRU(
            input_size=self.enc_in,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )
        
        # 保持分类器结构不变
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 3*self.hidden_size),
            self.relu,
            # nn.Linear(2*self.hidden_size, 3*self.hidden_size),
            # self.relu,
            nn.Linear(3*self.hidden_size, 4),
            self.relu,
            self.dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, 2*self.hidden_size),
            self.relu,
            self.dropout,
            nn.Linear(2*self.hidden_size, 4),
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # GRU前向传播（自动初始化隐藏状态）
        out, _ = self.gru1(x_enc)  # 输出形状: [batch, seq_len, hidden_size]
        state, _ = self.gru2(x_enc)
        
        # 保持激活和dropout处理一致
        out = self.relu(out)
        out = self.dropout(out)
        state = self.relu(state)
        state = self.dropout(state)
        
        # 保持最后时刻特征提取方式
        state = self.classifier(state[:, -1:, :])  # 取最后一个时间步
        out = self.fc(out[:, -1:, :])
        
        return out, state