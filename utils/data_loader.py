import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from config import Config


class SliceDataset(Dataset):
    def __init__(self, file_path, window_size=5):
        self.df = pd.read_csv(file_path)
        self.window_size = window_size
        self.slice_columns = ['iot', 'video', 'web']
        self.data = self.df[self.slice_columns].values.astype(np.float32)

    def __len__(self):
        return len(self.df) - self.window_size-1

    def __getitem__(self, idx):
        # 输入：多个切片的窗口数据 [window_size, num_slices]
        # 输出：下一时刻所有切片的需求 [num_slices]
        # 使用归一化后的数据
        hist = self.data[idx:idx + self.window_size] # [window*3]
        target = self.data[idx + self.window_size]  # [3]
        return torch.from_numpy(hist), torch.from_numpy(target)
if __name__ == '__main__':
    # 加载多切片数据集
    dataset = SliceDataset('../data/network_slices.csv', Config.window_size)
    loader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)

    for hist, target in loader:
        print(hist, target)
        break
