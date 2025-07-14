import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


class PowerDataset(torch.utils.data.Dataset):
    def __init__(self, features, targets, input_len, pred_len):
        self.X = features
        self.y = targets
        self.input_len = input_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.X) - self.input_len - self.pred_len + 1

    def __getitem__(self, idx):
        x_seq = self.X[idx:idx + self.input_len]    # shape: [input_len, F]
        y_seq = self.y[idx + self.input_len:idx + self.input_len + self.pred_len]  # shape: [pred_len, 1]
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_seq, dtype=torch.float32)


def aggregate_daily(df):
    """
    将分钟级别数据按天聚合为日级别数据。
    """
    df = df.set_index("DateTime").sort_index()

    # 以天为单位聚合：对功率类取平均，分表和降雨类取总和
    # 重新计算 sub_metering_remainder
    agg_df = df
    agg_df["sub_metering_remainder"] = (agg_df["Global_active_power"] * 1000 / 60) - (
        agg_df["Sub_metering_1"] + agg_df["Sub_metering_2"] + agg_df["Sub_metering_3"]
    )
    agg_df = agg_df.resample("1D").agg({
        'Global_active_power': 'mean',
        'Global_reactive_power': 'mean',
        'Voltage': 'mean',
        'Global_intensity': 'mean',
        'Sub_metering_1': 'sum',
        'Sub_metering_2': 'sum',
        'Sub_metering_3': 'sum',
        'sub_metering_remainder': 'sum',
        'RR': 'sum',
        'NBJRR1': 'sum',
        'NBJRR5': 'sum',
        'NBJRR10': 'sum',
        'NBJBROU': 'sum'
    })
    # print(agg_df.head())
    # 保存agg_df.head
    # agg_df.to_csv("agg_df.csv")

    return agg_df.dropna()


def load_and_preprocess(csv_path, normalize=True):
    df = pd.read_csv(csv_path, parse_dates=["DateTime"])
    df = df.dropna()
    df = df.astype({col: np.float32 for col in df.columns if col != "DateTime"})

    # 聚合为每日数据
    daily_df = aggregate_daily(df)

    feature_cols = [
        'Global_reactive_power', 'Voltage', 'Global_intensity',
        'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3',
        'RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU',
        'sub_metering_remainder'
    ]
    target_col = 'Global_active_power'

    features = daily_df[feature_cols].values
    targets = daily_df[[target_col]].values

    # 进行归一化，可能会出现负数
    if normalize:
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        features = scaler_x.fit_transform(features)
        targets = scaler_y.fit_transform(targets)
    else:
        scaler_x = scaler_y = None

    return features, targets, scaler_x, scaler_y


def build_dataloader(features, targets, input_days=90, pred_days=90, batch_size=32, shuffle=False):
    dataset = PowerDataset(features, targets, input_len=input_days, pred_len=pred_days)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


if __name__ == "__main__":
    # 示例用法
    train_csv = "/data/hyt/ml/train.csv"
    test_csv = "/data/hyt/ml/test.csv"

    # 加载并处理训练集和测试集，日粒度数据，标准化
    features_train, targets_train, sx, sy = load_and_preprocess(train_csv)
    features_test, targets_test, _, _ = load_and_preprocess(test_csv)

    # 构建短期预测 (90天 -> 90天)
    train_loader = build_dataloader(features_train, targets_train, input_days=90, pred_days=90, batch_size=32, shuffle=True)
    test_loader = build_dataloader(features_test, targets_test, input_days=90, pred_days=90, batch_size=32, shuffle=False)

    # 构建长期预测 (90天 -> 365天)
    test_loader_long = build_dataloader(features_test, targets_test, input_days=90, pred_days=365, batch_size=8, shuffle=False)

    # 查看一个 batch 的形状
    for xb, yb in train_loader:
        print("📦 短期预测输入形状:", xb.shape)   # [B, 90, F]
        print("📦 短期预测目标形状:", yb.shape)  # [B, 90, 1]
        # 查看一个 batch 的数据
        print("📦 短期预测输入数据:")
        print(xb)
        print("📦 短期预测目标数据:")
        print(yb)
        break
    for xb, yb in test_loader:
        print("📦 短期预测输入形状:", xb.shape)   # [B, 90, F]
        print("📦 短期预测目标形状:", yb.shape)  # [B, 90, 1]
        break
