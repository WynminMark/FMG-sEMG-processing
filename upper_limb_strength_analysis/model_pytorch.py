import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn import preprocessing




# 自定义数据集类
class MyDataset(Dataset):
    def __init__(self, X_data, Y_data):
        """
        初始化数据集，X_data 和 Y_data 是两个列表或数组
        X_data: 输入特征
        Y_data: 目标标签
        """
        self.X_data = X_data
        self.Y_data = Y_data

    def __len__(self):
        """返回数据集的大小"""
        return len(self.X_data)

    def __getitem__(self, idx):
        """返回指定索引的数据"""
        x = torch.tensor(self.X_data[idx], dtype=torch.float32)  # 转换为 Tensor
        y = torch.tensor(self.Y_data[idx], dtype=torch.float32)
        return x, y

# 自定义模型类
class Model(nn.Module):
    def __init__(self, n_input,n_hidden,n_output):
        super(Model, self).__init__()
        self.hidden1 = nn.Linear(n_input, n_hidden)
        self.hidden2 = nn.Linear(n_hidden, n_hidden)
        self.predict = nn.Linear(n_hidden, n_output)

        pass
    
    def forward(self, x):
        out = self.hidden1(x)
        out = F.relu(out)
        out = self.hidden2(out)
        out = F.sigmoid(out)
        out =self.predict(out)
        pass

    pass

if __name__ == '__main__':
    # 读数据
    data_df = pd.read_csv(r"/home/weimy/data/iFEMG_dataset/upper_limb/iFEMG/iFEMG_extension_abs_normed_all.csv", index_col = None, header = 0)
    # 数据预处理, 标准化
    subject_feature_columes = ['subject_info_height', 'subject_info_weight', 'subject_info_age', 'subject_info_label']
    data_df[subject_feature_columes] = preprocessing.StandardScaler().fit_transform(data_df[subject_feature_columes])

    columns_to_scale = ['subject_info_height',
                    'subject_info_weight',
                    'subject_info_age',
                    'bicps_br_initial_pressure_ave',
                    'bicps_br_FMG',
                    'bicps_br_mav',
                    'bicps_br_rms',
                    'bicps_br_wave_length',
                    'bicps_br_zero_crossing',
                    'bicps_br_slope_sign_change',
                    'bicps_br_mean_freq',
                    'bicps_br_mean_power_freq',
                    'tricps_br_medial_initial_pressure_ave',
                    'tricps_br_medial_FMG',
                    'tricps_br_medial_mav',
                    'tricps_br_medial_rms',
                    'tricps_br_medial_wave_length',
                    'tricps_br_medial_zero_crossing',
                    'tricps_br_medial_slope_sign_change',
                    'tricps_br_medial_mean_freq',
                    'tricps_br_medial_mean_power_freq',
                    'tricps_br_lateral_initial_pressure_ave',
                    'tricps_br_lateral_FMG',
                    'tricps_br_lateral_mav',
                    'tricps_br_lateral_rms',
                    'tricps_br_lateral_wave_length',
                    'tricps_br_lateral_zero_crossing',
                    'tricps_br_lateral_slope_sign_change',
                    'tricps_br_lateral_mean_freq',
                    'tricps_br_lateral_mean_power_freq']
    print(data_df.shape)
    #dataset = MyDataset(data_df.loc[:, columns_to_scale].values, data_df.loc[:, 'subject_info_label'].values)
    #dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    X = torch.tensor(data_df.loc[:, columns_to_scale].values, dtype=torch.float32)
    Y = torch.tensor(data_df.loc[:, 'subject_info_label'].values, dtype=torch.float32)
    # ==================== 数据准备完成 ====================
    # 模型
    model = Model(30, 90, 1)
    # 损失函数（均方误差）
    criterion = nn.MSELoss()
    # 优化器（使用 SGD 或 Adam）
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 学习率设置为0.01

    # 指定GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 将模型移动到设备
    model.to(device)

    # 训练模型
    model.train()  # 设置模型为训练模式
    num_epochs = 1000  # 训练 1000 轮
    for epoch in range(num_epochs):

        X, Y = X.to(device), Y.to(device)  # 将数据移动到设备
        # 前向传播
        predictions = model(X)  # 模型输出预测值
        if predictions is not None:
            predictions = predictions.squeeze()
            #print("Predictions:", predictions)
            loss = criterion(predictions, Y)
            #print(f'Loss: {loss.item():.4f}')
        else:
            continue
            #print("Error: model(X) returned None")

        # 反向传播
        optimizer.zero_grad()  # 清空之前的梯度
        loss.backward()  # 计算梯度
        optimizer.step()  # 更新模型参数

        # 打印损失
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/1000], Loss: {loss.item():.4f}')


    # 查看训练后的权重和偏置
    #print(f'Predicted weight: {model.linear.weight.data.numpy()}')
    #print(f'Predicted bias: {model.linear.bias.data.numpy()}')


    pass
