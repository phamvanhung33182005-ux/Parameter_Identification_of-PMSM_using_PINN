# ============================================================
# FILE: src/models/mlp.py
# ============================================================
#
# Mục đích:
#     Định nghĩa kiến trúc MLP baseline cho bài toán
#     nhận dạng tham số PMSM.
#
# Input:
#     5 features
#
# Output:
#     3 PMSM parameters
#
# Kiến trúc:
#     Input(5)
#        ↓
#     Linear(5, 64)
#        ↓
#     ReLU
#        ↓
#     Linear(64, 64)
#        ↓
#     ReLU
#        ↓
#     Linear(64, 3)
#
# Lưu ý:
#     File này CHỈ định nghĩa model.
#
#     Chưa thực hiện:
#     - training
#     - loss
#     - optimizer
#     - validation
#     - tuning
#     - checkpoint
# ============================================================


# Import module neural network của PyTorch.
import torch.nn as nn


# Định nghĩa class BaselineMLP.
class BaselineMLP(nn.Module):

    # Hàm khởi tạo model.
    def __init__(
        self,
        input_size=5,
        hidden_size=64,
        output_size=3
    ):

        # Khởi tạo lớp cha nn.Module.
        super().__init__()

        # Xây dựng kiến trúc mạng MLP.
        self.network = nn.Sequential(

            # Fully-connected layer đầu tiên.
            # 5 input features → 64 neurons.
            nn.Linear(
                input_size,
                hidden_size
            ),

            # Hàm kích hoạt ReLU.
            nn.ReLU(),

            # Fully-connected hidden layer thứ hai.
            # 64 neurons → 64 neurons.
            nn.Linear(
                hidden_size,
                hidden_size
            ),

            # Hàm kích hoạt ReLU.
            nn.ReLU(),

            # Output layer.
            # 64 neurons → 3 outputs.
            nn.Linear(
                hidden_size,
                output_size
            )
        )

    # Hàm forward xác định cách dữ liệu đi qua mạng.
    def forward(self, x):

        # Đưa input x qua toàn bộ neural network.
        return self.network(x)
