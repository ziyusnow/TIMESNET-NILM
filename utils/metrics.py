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
    valid_indices = np.abs(true) > 0.1
    vali_pred_values = pred[valid_indices]
    vali_true_values = true[valid_indices]
    mae = MAE(vali_pred_values, vali_true_values)
    mse = MSE(vali_pred_values, vali_true_values)
    rmse = RMSE(vali_pred_values, vali_true_values)
    mape = MAPE(vali_pred_values, vali_true_values)
    mspe = MSPE(vali_pred_values, vali_true_values)
    # mae = MAE(pred, true)
    # mse = MSE(pred, true)
    # rmse = RMSE(pred, true)
    # mape = MAPE(pred, true)
    # mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe
