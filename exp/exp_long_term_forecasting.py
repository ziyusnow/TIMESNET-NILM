from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual,compute_loss_weights,FocalLoss
from utils.metrics import metric
from utils.status_metrics import status_metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw,accelerated_dtw
from utils.augmentation import run_augmentation,run_augmentation_single
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from logger_factory import setup_logger
warnings.filterwarnings('ignore')
logger = setup_logger('tool_test')

class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.writer = SummaryWriter(log_dir='./logs')  # 创建一个 TensorBoard 的日志记录器

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate,weight_decay=self.args.weight_decay)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss().to(self.device)
        classification_loss = nn.BCEWithLogitsLoss().to(self.device)
        return criterion,classification_loss

    def vali(self, vali_data, vali_loader, criterion,classification_loss):
        total_loss = []
        vali_y_true = []
        vali_y_pred = []
        vali_s_true = []
        vali_s_pred = []                
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_s,batch_x_mark, batch_y_mark) in tqdm(enumerate(vali_loader),desc='vali',total=len(vali_loader)):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_s = batch_s.float().to(self.device) 
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -1:, :]).float()
                dec_inp = torch.cat([batch_y[:, :, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs,states = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -1:, f_dim:]
                states = states[:, -1:, f_dim:]
                states=torch.sigmoid(states)
                states = (states >= 0.6).float()
                batch_y = batch_y[:, -1:, f_dim:].to(self.device)
                outputs=states*outputs
                re_loss = criterion(outputs, batch_y).detach().cpu()
                cla_loss= classification_loss(states,batch_s).detach().cpu()
                loss=re_loss+cla_loss*0.2
                vali_y_true.append(batch_y.detach().cpu().numpy())
                vali_y_pred.append(outputs.detach().cpu().numpy())
                vali_s_true.append(batch_s.detach().cpu().numpy())
                vali_s_pred.append(states.detach().cpu().numpy())                                   
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        vali_y_true = np.concatenate(vali_y_true, axis=0)
        vali_y_pred = np.concatenate(vali_y_pred, axis=0)
        vali_s_true = np.concatenate(vali_s_true, axis=0)
        vali_s_pred = np.concatenate(vali_s_pred, axis=0)        
        vali_y_pred = vali_y_pred.reshape(-1, 4)
        vali_y_true = vali_y_true.reshape(-1, 4)
        vali_s_pred = vali_s_pred.reshape(-1, 4)
        vali_s_true = vali_s_true.reshape(-1, 4)        
        vali_y_true=vali_y_true*vali_data.dif_value[1:]+vali_data.min_value[1:]
        vali_y_pred=vali_y_pred*vali_data.dif_value[1:]+vali_data.min_value[1:]
        self.model.train()
        return total_loss,vali_y_true,vali_y_pred,vali_s_true,vali_s_pred

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion,classification_loss = self._select_criterion()   #MSE

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            all_y_true = []
            all_y_pred = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_s,batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader),total=len(train_loader)):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device) #[4,64,4]
                batch_y = batch_y.float().to(self.device) #[4,1,4]
                batch_s = batch_s.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -1:, :]).float()
                dec_inp = torch.cat([batch_y[:, :, :], dec_inp], dim=1).float().to(self.device) #[4,2,4]

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -1:, f_dim:]
                        batch_y = batch_y[:, -1:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs,states = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)#[4,64,4]

                    f_dim = -1 if self.args.features == 'MS' else 0
                    states = states[:, -1:, f_dim:]
                    batch_s = batch_s[:, -1:, f_dim:].to(self.device) 
                    cla_loss=classification_loss(states,batch_s)
                    states=torch.sigmoid(states)
                    states = (states >= 0.6).float()
                    outputs = outputs[:, -1:, f_dim:]#[4,1,4]
                    outputs=outputs*states
                    batch_y = batch_y[:, -1:, f_dim:].to(self.device)
                    all_y_true.append(batch_y.detach().cpu().numpy())
                    all_y_pred.append(outputs.detach().cpu().numpy())  
                    re_loss = criterion(outputs, batch_y)
                    loss=re_loss+cla_loss*0.2

                    train_loss.append(loss.item())

                if (i + 1) % 5000 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
            all_y_true = np.concatenate(all_y_true, axis=0)
            all_y_pred = np.concatenate(all_y_pred, axis=0)
            all_y_pred = all_y_pred.reshape(-1, 4)
            all_y_true = all_y_true.reshape(-1, 4)
            all_y_true=all_y_true*train_data.dif_value[1:]+train_data.min_value[1:]
            all_y_pred=all_y_pred*train_data.dif_value[1:]+train_data.min_value[1:]
            mae, mse, rmse, mape, mspe,mae1 = metric(all_y_true, all_y_pred)
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            logger.info('Train Loss | mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}'.format(mse, mae, rmse, mape, mspe))
            vali_loss,vali_y_true,vali_y_pred,vali_s_true,vali_s_pred= self.vali(vali_data, vali_loader, criterion,classification_loss)
            self.writer.add_scalar('Loss/Validation', vali_loss, epoch)
            test_loss,test_y_true,test_y_pred,vali_s_true,vali_s_pred = self.vali(test_data, test_loader, criterion,classification_loss)
            mae, mse, rmse, mape, mspe,mae1 = metric(vali_y_true, vali_y_pred)
            logger.info('Vali Loss | mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}'.format(mse, mae, rmse, mape, mspe))
            f1, precision, recall, accuracy = status_metric(vali_s_pred,vali_s_true)
            logger.info('Vali Loss |  f1:{}, precision:{}, recall:{}, accuracy:{}'.format(f1, precision, recall, accuracy))

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            logger.info("Learning Rate: {0}, Batch Size: {1}, Optimizer: {2}".format(
                model_optim.param_groups[0]['lr'], self.args.batch_size, type(model_optim).__name__))
            logger.info("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args,self.writer)
        self.writer.close()  # 关闭 TensorBoard 记录器
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return vali_loss,self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        s_hat=[]
        s_true=[]
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_s,batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader),desc='test',total=len(test_loader)):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_s =batch_s.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -1:, :]).float()
                dec_inp = torch.cat([batch_y[:, :, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs,states = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -1:, :]
                states = states[:, -1:, :]
                states=torch.sigmoid(states)
                states = (states >= 0.6).float()
                batch_y = batch_y[:, -1:, :]
                outputs=outputs*states
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                states=states.detach().cpu().numpy()
                batch_s=batch_s.detach().cpu().numpy()
                batch_x = batch_x.detach().cpu().numpy()
                outputs=np.maximum(outputs,0)
                if test_data.scale and self.args.inverse:
                    # batch_x_sig = batch_x[:,:,:1]
                    # outputs = np.concatenate((batch_x_sig, outputs), axis=1)
                    # batch_y = np.concatenate((batch_x_sig, batch_y), axis=1)

                    shape = outputs.shape #(4,1,4)
                    outputs=outputs*test_data.dif_value[1:]+test_data.min_value[1:]
                    batch_y=batch_y*test_data.dif_value[1:]+test_data.min_value[1:]
                    batch_x=batch_x*test_data.dif_value[0]+test_data.min_value[0]
                    #outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    #batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]
                batch_x=batch_x[:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                s_hat.append(states)
                s_true.append(batch_s)
                # if i % 20 == 0:
                #     #input = batch_x[:,:,0].detach().cpu().numpy()
                #     input = batch_x[:,:,0]
                #     if test_data.scale and self.args.inverse:
                #         shape = input.shape
                #         input = input*test_data.std_value[0]+test_data.mean_value[0]
                    
                    #需要画图更新    
                    # gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    # pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    # visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf')) 


        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        s_hat = np.concatenate(s_hat, axis=0)
        s_true = np.concatenate(s_true, axis=0)        
        preds = preds.reshape(-1, preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-1])
        s_hat = s_hat.reshape(-1, s_hat.shape[-1])
        s_true = s_true.reshape(-1, s_true.shape[-1])
        #preds[preds < 0] = 0 

        for i in range(4):
          pd=preds[:,i]
          gt=trues[:,i]          
          visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
          sh=s_hat[:,i]
          st=s_true[:,i]
          visual(st, sh, os.path.join(folder_path, str(i) + 'state.pdf'))
        print('test shape:', preds.shape, trues.shape)
        

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1,1)
                y = trues[i].reshape(-1,1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = -999
            

        mae, mse, rmse, mape, mspe,mae1= metric(preds, trues)
        print('mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{},mae1:{}, dtw:{}'.format(mse, mae, rmse, mape, mspe,mae1,dtw))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}, mae1:{},dtw:{}'.format(mse, mae, rmse, mape, mspe,mae1,dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        f1, precision, recall, accuracy = status_metric(s_hat, s_true)
        print('f1:{}, precision:{}, recall:{}, accuracy:{}'.format(f1, precision, recall, accuracy))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('f1:{}, precision:{}, recall:{}, accuracy:{}'.format(f1, precision, recall, accuracy))
        f.write('\n')
        f.write('\n')
        f.close()
        logger.info('mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}'.format(mse, mae, rmse, mape, mspe))
        logger.info('f1:{}, precision:{}, recall:{}, accuracy:{}'.format(f1, precision, recall, accuracy))

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        np.save(folder_path + 'status_metrics.npy', np.array([f1, precision, recall, accuracy]))
        np.save(folder_path + 's_hat.npy', s_hat)
        np.save(folder_path + 's_true.npy', s_true)




        return
