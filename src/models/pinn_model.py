# -*- coding: utf-8 -*-
"""
Mô hình Physics-Informed Neural Network (PINN) xác định tham số PMSM
Tác giả: Phạm Văn Hùng (phamvanhung33182005-ux)
"""

import torch
import torch.nn as nn
from src.models.baselineMLP import BaselineMLP

class PMSM_PINN(nn.Module):
    def __init__(self, phi_m=0.23890939):
        super(PMSM_PINN, self).__init__()

        # Sử dụng MLP làm mạng suy luận tham số cốt lõi
        self.backbone = BaselineMLP(input_dim=7, output_dim=3)
        # Giá trị từ thông nam châm cố định theo yêu cầu kỹ thuật
        self.phi_m = phi_m

    def forward(self, x):
        # Dự đoán các tham số Rs, Ld, Lq
        params = self.backbone(x)
        return params

    def compute_physics_residual(self, x, params):
        """
        Tính toán phần dư phương trình vi phân sụt áp d-q động cơ PMSM
        u_d = Rs * i_d + Ld * (di_d/dt) - omega_e * Lq * i_q
        u_q = Rs * i_q + Lq * (di_q/dt) + omega_e * Ld * i_d + omega_e * phi_m
        """
        # Tách các thành phần đặc trưng đầu vào
        # x: [sector, V_alpha, V_beta, V_dc, i_d, i_q, omega_e]
        i_d = x[:, 4]
        i_q = x[:, 5]
        omega_e = x[:, 6]

        # Tham số ước lượng từ mạng
        Rs_pred = params[:, 0]
        Ld_pred = params[:, 1]
        Lq_pred = params[:, 2]

        # Tái tạo điện áp trên hệ tọa độ d-q ở trạng thái xác lập (di/dt = 0)
        u_d_est = Rs_pred * i_d - omega_e * Lq_pred * i_q
        u_q_est = Rs_pred * i_q + omega_e * Ld_pred * i_d + omega_e * self.phi_m

        return u_d_est, u_q_est
