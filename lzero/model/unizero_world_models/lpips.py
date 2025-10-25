"""
Modified from https://github.com/CompVis/taming-transformers
"""

import hashlib
import os
from collections import namedtuple
from pathlib import Path

import requests
import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm

# ==================================================================================
# ================================ 核心修改部分 ====================================
# ==================================================================================
# 在导入 torch 和 torchvision 之后，但在实例化任何模型之前，设置 TORCH_HOME 环境变量。
# 这会告诉 PyTorch 将所有通过 torch.hub 下载的模型（包括 torchvision.models 中的预训练模型）
# 存放到您指定的目录下。
# PyTorch 会自动在此目录下创建 'hub/checkpoints' 子文件夹。
custom_torch_home = "/mnt/shared-storage-user/puyuan/code_20250828/LightZero/tokenizer_pretrained_vgg"
os.environ['TORCH_HOME'] = custom_torch_home
# 确保目录存在，虽然 torch.hub 也会尝试创建，但提前创建更稳妥
os.makedirs(os.path.join(custom_torch_home, 'hub', 'checkpoints'), exist_ok=True)
# ==================================================================================
# ==================================================================================


class LPIPS(nn.Module):
    # Learned perceptual metric
    def __init__(self, use_dropout: bool = True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]  # vg16 features

        # Comment out the following line if you don't need perceptual loss
        # 现在，这一行将自动使用 TORCH_HOME 指定的路径
        self.net = vgg16(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.load_from_pretrained()
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self) -> None:
        # 这一部分您已经修改正确，它用于加载 LPIPS 的线性层权重 (vgg.pth)
        # 我们让它和 TORCH_HOME 使用相同的根目录，以保持一致性。
        ckpt = get_ckpt_path(name="vgg_lpips", root=custom_torch_home)
        self.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)
        print(f"Loaded LPIPS pretrained weights from: {ckpt}")

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [spatial_average(lins[kk].model(diffs[kk]), keepdim=True) for kk in range(len(self.chns))]
        val = res[0]
        for i in range(1, len(self.chns)):
            val += res[i]
        return val


class ScalingLayer(nn.Module):
    def __init__(self) -> None:
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """
    def __init__(self, chn_in: int, chn_out: int = 1, use_dropout: bool = False) -> None:
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad: bool = False, pretrained: bool = True) -> None:
        super(vgg16, self).__init__()
        # 由于设置了 TORCH_HOME，这里的 pretrained=True 会在指定目录中查找或下载模型
        print("Loading vgg16 backbone...")
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        print("vgg16 backbone loaded.")
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


def normalize_tensor(x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def spatial_average(x: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
    return x.mean([2, 3], keepdim=keepdim)


# ********************************************************************
# *************** Utilities to download pretrained vgg ***************
# ********************************************************************


URL_MAP = {
    "vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"
}


CKPT_MAP = {
    "vgg_lpips": "vgg.pth"
}


MD5_MAP = {
    "vgg_lpips": "d507d7349b931f0638a25a48a722f98a"
}


def download(url: str, local_path: str, chunk_size: int = 1024) -> None:
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)


def md5_hash(path: str) -> str:
    with open(path, "rb") as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()


def get_ckpt_path(name: str, root: str, check: bool = False) -> str:
    assert name in URL_MAP
    # 这个函数现在只为 vgg.pth 服务，路径是正确的
    path = os.path.join(root, CKPT_MAP[name])
    if not os.path.exists(path) or (check and not md5_hash(path) == MD5_MAP[name]):
        print("Downloading {} model from {} to {}".format(name, URL_MAP[name], path))
        download(URL_MAP[name], path)
        md5 = md5_hash(path)
        assert md5 == MD5_MAP[name], md5
    return path

# =======================
# =====  运行示例  ======
# =======================
if __name__ == '__main__':
    print(f"PyTorch Hub directory set to: {os.environ['TORCH_HOME']}")
    
    # 第一次运行时，你会看到两个下载过程：
    # 1. 下载 vgg16-397923af.pth 到 /mnt/shared-storage-user/puyuan/code_20250828/LightZero/tokenizer_pretrained_vgg/hub/checkpoints/
    # 2. 下载 vgg.pth 到 /mnt/shared-storage-user/puyuan/code_20250828/LightZero/tokenizer_pretrained_vgg/
    # 之后再次运行，将不会有任何下载提示，直接从指定目录加载。
    
    print("\nInitializing LPIPS model...")
    model = LPIPS()
    print("\nLPIPS model initialized successfully.")