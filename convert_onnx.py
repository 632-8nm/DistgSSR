import torch
import os
from model import Net  # 根据你的目录结构修改
import sys

# 参数
angRes = 5
factor = 2  # 放大倍数
H, W = 128, 128  # 中心视角图像尺寸
device = torch.device("cpu")

# 1. 加载模型
model = Net(angRes, factor).to(device)
model.eval()
ckpt_path = "log/DistgSSR_2xSR_5x5.pth.tar"
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt['state_dict'])

# 2. 构造输入
input_tensor = torch.rand(1, 1, H * angRes, W * angRes).to(device)  # [1,1,H*u,W*v]

# 3. 导出 ONNX
save_path = ckpt_path.replace(".pth.tar", ".onnx")
torch.onnx.export(
    model,
    input_tensor,
    save_path,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {2: 'H_in', 3: 'W_in'},  # 支持动态分辨率（可选）
        'output': {2: 'H_out', 3: 'W_out'}
    }
)
print(f"ONNX model exported to: {save_path}")
