from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
import seaborn as sns
from pathlib import Path
from layers.Conv_Blocks import Inception_Block_V1
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd

class VisualConfig:
    def __init__(self):
        self.save_dir = Path('./visual_train1_results')
        self.cmap = self._create_blue_white_cmap()
        self.active = False
        self.batch_num = 0
        self.current_epoch = 0
        self.layer_index = 0
        
    def _create_blue_white_cmap(self):
        """创建白到蓝渐变色谱"""
        return LinearSegmentedColormap.from_list(
            'white_blue',
            [(1, 1, 1), (0.02, 0.16, 0.47)],  # 白到深蓝渐变
            N=256
        )
    
    def get_batch_path(self,batch_num, layer, k_index, period):
        # 示例路径: visual_results/batch_000/layer_0
        #save_dir = self.save_dir /f"batch_{batch_num:03d}" / f"layer_{layer}"
        save_dir = self.save_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 文件名示例: k0_p4.png
        return save_dir / f"k{k_index}_p{period}.png"
    
    def get_data_path(self, batch_num, layer_index, k_index, period):
        # 示例路径: visual_results/batch_000/layer_0/k0_p4.csv
        save_dir = self.save_dir / f"batch_{batch_num:03d}" / f"layer_{layer_index}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 文件名示例: k0_p4.csv
        return save_dir / f"k{k_index}_p{period}.csv"


def FFT_for_Period(x, k=2):
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]

class TimesBlock(nn.Module):
    def __init__(self, configs,visual_config,layer_index):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        self.visual_config = visual_config 
        self.layer_index = layer_index
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )


    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
                        # 可视化堆叠后的向量
            # 新增可视化调用
            if self.visual_config.active and period>70 and period<100:  # 使用配置的激活状态  
                self._visualize_components(out, period, i,self.visual_config.batch_num,self.layer_index)
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res
        
    def _visualize_components(self, tensor, period, k_index,batch,layer):
        """可视化核心方法"""
        # 确保在CPU上操作
        tensor = tensor.detach().cpu().float()
        
        # 自动选择可视化位置
        batch_idx = 0  # 只看第一个样本
        channel_idx = min(1, tensor.shape[1]-1)  # 选择第一个有效通道
        
        data = tensor[batch_idx, channel_idx]  # [num_segments, period]
        
        # 创建画布
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 2D热力图
        sns.heatmap(data.numpy(), ax=axes[0], cmap=self.visual_config.cmap, 
                    xticklabels=10, yticklabels=10)
        axes[0].set_title(f'2D Structure\nPeriod: {period} | K: {k_index+1}\nBatch: {batch} | Layer: {layer}')
        axes[0].set_xlabel('Time Position in Period')
        axes[0].set_ylabel('Segment Index')
        
        # 时域波形图
        time_axis = torch.arange(data.shape[0]*data.shape[1]) / data.shape[1]
        axes[1].plot(time_axis, data.flatten().numpy())
        axes[1].set_title('Time Domain View')
        axes[1].set_xlabel('Normalized Time')
        axes[1].set_ylabel('Value')
        axes[1].grid(True)
        
        # 自动保存图像
        save_path = self.visual_config.get_batch_path(self.visual_config.batch_num, self.layer_index, k_index, period)
        plt.savefig(save_path)
        plt.close()  # 防止内存泄漏

        # 保存数据
        data_path = self.visual_config.get_data_path(self.visual_config.batch_num, self.layer_index, k_index, period)
        df = pd.DataFrame(data.numpy())
        df.to_csv(data_path, index=False)


class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs,visual_config):
        super(Model, self).__init__()
        self.configs = configs
        self.visual_config = visual_config
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.model = nn.ModuleList([
            TimesBlock(configs, self.visual_config, layer_index=i)
            for i in range(configs.e_layers)  # i从0开始递增
        ])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.num=16


            
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.act = nn.GELU()
            self.dropout = nn.Dropout(configs.dropout)
            self.predict_linear1 = nn.Linear(
                self.seq_len, self.pred_len + self.seq_len)
            self.predict_linear2 = nn.Linear(
                self.pred_len + self.seq_len, 1)
            self.projection = nn.Sequential(
            nn.Linear(configs.d_model, self.num*configs.d_model, bias=True),           
            self.act,
            self.dropout,
            nn.Linear(self.num*configs.d_model, configs.c_out, bias=True),
            )
            self.classifier = nn.Sequential(
            nn.Linear(configs.d_model , self.num*configs.d_model ),
            self.act,
            self.dropout,
            nn.Linear(self.num*configs.d_model , configs.c_out)
            )


        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)
                
        

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        #if not self.visual_config.is_training:
        self.visual_config.active = True        
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means #[4,64,4]
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]  #[4,64,16]
        enc_out = self.predict_linear1(enc_out.permute(0, 2, 1)).permute(0, 2, 1) # align temporal dimension
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        enc_out = self.act(enc_out)
        enc_out = self.dropout(enc_out)
        fc_out = self.projection(enc_out) 
        dec_out=self.predict_linear2(fc_out.permute(0, 2, 1)).permute(0, 2, 1)         

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, 1, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, 1, 1))
        #state = self.classifier(enc_out[:,-1:,:])
        self.visual_config.active = False     
        return dec_out#state #[B,1,d]

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        #output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        #output = output.reshape(output.shape[0],1, -1)
        output = self.classifier(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            state_out = self.classification(x_enc, x_mark_enc)
            return dec_out[:, -self.pred_len:, :],state_out[:, -1:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
