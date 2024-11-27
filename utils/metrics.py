import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(true - pred))


def MSE(pred, true):
    return np.mean((true - pred) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((true - pred) / true))


def MSPE(pred, true):
    return np.mean(np.square((true - pred) / true))


def metric(pred, true):
    # 初始化存储每列指标的列表
    mae_list, mse_list, rmse_list, mape_list, mspe_list = [], [], [], [], []
    
    # 获取列数（设备数量）
    n_devices = pred.shape[1]
    
    # 对每一列分别计算指标
    for i in range(n_devices):
        # 获取当前列的预测值和真实值
        pred_col = pred[:, i]
        true_col = true[:, i]
        
        # 应用有效值过滤条件
        valid_indices = np.abs(true_col) > 0.1
        vali_pred_values = pred_col[valid_indices]
        vali_true_values = true_col[valid_indices]
        
        # 计算各项指标
        mae = MAE(vali_pred_values, vali_true_values)
        mse = MSE(vali_pred_values, vali_true_values)
        rmse = RMSE(vali_pred_values, vali_true_values)
        mape = MAPE(vali_pred_values, vali_true_values)
        mspe = MSPE(vali_pred_values, vali_true_values)
        
        # 将结果添加到对应的列表中
        mae_list.append(mae)
        mse_list.append(mse)
        rmse_list.append(rmse)
        mape_list.append(mape)
        mspe_list.append(mspe)
    
    # 将列表转换为numpy数组
    mae_array = np.array(mae_list)
    mse_array = np.array(mse_list)
    rmse_array = np.array(rmse_list)
    mape_array = np.array(mape_list)
    mspe_array = np.array(mspe_list)
    
    return mae_array, mse_array, rmse_array, mape_array, mspe_array
