import numpy as np

from utils.metrics import metric
# 读取保存的预测值和真实值的 numpy 文件
pred = np.load('/gemini/code/results/12061918_inverse_test_long_term_forecast_LSTM_custom_ftM_sl64_ll32_pl0_dm8_nh1_el1_dl1_df16_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0/pred.npy')  # 假设你的预测值保存在 pred.npy 文件中
true = np.load('/gemini/code/results/12061918_inverse_test_long_term_forecast_LSTM_custom_ftM_sl64_ll32_pl0_dm8_nh1_el1_dl1_df16_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0/true.npy')  # 假设你的真实值保存在 true.npy 文件中


# 使用 scikit-learn 的 mean_absolute_error

mae, mse, rmse, mape, mspe,mae1= metric(pred, true)
print('mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{},mae1:{}'.format(mse, mae, rmse, mape, mspe,mae1))