# -*- coding: utf-8 -*-
"""
Mô hình Physics-Informed Neural Network (PINN) xác định tham số PMSM
Tác giả: Phạm Văn Hùng (phamvanhung33182005-ux)
"""

import torch
import torch.nn as nn
import numpy as np
from src.models.baselineMLP import BaselineMLP

class PMSM_PINN(nn.Module):
    def __init__(self, phi_m=0.23890939, pole_pairs=4):
        super(PMSM_PINN, self).__init__()

        # Truyền rõ ràng input_dim=5 và output_dim=3
        self.backbone = BaselineMLP(input_dim=5, output_dim=3)
        self.phi_m = phi_m
        self.p = pole_pairs

    def forward(self, x):
        params = self.backbone(x)
        return params

    def compute_physics_residual(self, x, params):
        # x: [ud, uq, Id, Iq, nm]
        i_d = x[:, 2]
        i_q = x[:, 3]
        nm  = x[:, 4]

        # Quy đổi tốc độ cơ (rpm) sang tốc độ góc điện omega_e (rad/s)
        omega_e = nm * (2 * np.pi / 60) * self.p

        Rs_pred = params[:, 0]
        Ld_pred = params[:, 1]
        Lq_pred = params[:, 2]

        # Tái tạo điện áp d-q
        u_d_est = Rs_pred * i_d - omega_e * Lq_pred * i_q
        u_q_est = Rs_pred * i_q + omega_e * Ld_pred * i_d + omega_e * self.phi_m

        return u_d_est, u_q_est
