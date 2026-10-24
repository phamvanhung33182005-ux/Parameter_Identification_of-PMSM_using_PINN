# -*- coding: utf-8 -*-
"""
Module Quản lý Dữ liệu Động cơ PMSM cho Mạng PINN
Tác giả: Phạm Văn Hưng (phamvanhung33182005-ux)
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class PMSMDataset(Dataset):
    def __init__(self, csv_file_path):
        """
        Khởi tạo Dataset từ file CSV đã chuẩn hóa
        """
        self.data = pd.read_csv(csv_file_path)

        # Đặc trưng đầu vào (5 features: ud, uq, Id, Iq, nm)
        self.feature_cols = ['ud', 'uq', 'Id', 'Iq', 'nm']
        # Nhãn đầu ra cần ước lượng (3 targets: Rs, Ld, Lq)
        self.target_cols = ['Rs', 'Ld', 'Lq']

        self.X = torch.tensor(self.data[self.feature_cols].values, dtype=torch.float32)
        self.y = torch.tensor(self.data[self.target_cols].values, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
