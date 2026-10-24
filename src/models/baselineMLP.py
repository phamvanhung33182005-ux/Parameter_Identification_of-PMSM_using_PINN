# -*- coding: utf-8 -*-
"""
Mô hình Baseline MLP Thuần Dữ liệu cho Xác định Tham số PMSM
Tác giả: Phạm Văn Hùng (phamvanhung33182005-ux)
"""

import torch
import torch.nn as nn

class BaselineMLP(nn.Module):
    def __init__(self, input_dim=7, output_dim=3):
        super(BaselineMLP, self).__init__()

        # Mạng Perceptron đa lớp với hàm kích hoạt SiLU
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        # x: [batch_size, 7] -> outputs: [batch_size, 3] (Rs, Ld, Lq)
        return self.network(x)
