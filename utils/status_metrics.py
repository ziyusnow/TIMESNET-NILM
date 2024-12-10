from sklearn.metrics import f1_score, precision_score, recall_score,accuracy_score
import numpy as np

def status_metric(s_h, s_t):
    # 初始化存储每列指标的列表
    f1_list, precision_list, recall_list, accuracy_list = [], [], [], []
    
    # 获取列数（设备数量）
    n_devices = s_h.shape[1]
    
    # 对每一列分别计算指标
    for i in range(n_devices):
        # 获取当前列的预测值和真实值
        sh_col = s_h[:, i]
        st_col = s_t[:, i]
        
        # 计算各项指标
        f1 = f1_score(sh_col, st_col)
        precision = precision_score(sh_col, st_col)
        recall = recall_score(sh_col, st_col)
        accuracy = accuracy_score(sh_col, st_col)
        
        # 将结果添加到对应的列表中
        f1_list.append(f1)
        precision_list.append(precision)
        recall_list.append(recall)
        accuracy_list.append(accuracy)

    
    # 将列表转换为numpy数组
    f1_array = np.array(f1_list)
    precision_array = np.array(precision_list)
    recall_array = np.array(recall_list)
    accuracy_array = np.array(accuracy_list)
    
    return f1_array, precision_array, recall_array, accuracy_array