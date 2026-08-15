"""
Modified from https://github.com/CompVis/taming-transformers
"""

import hashlib
import os
import torch
import torch.nn as nn
from torchvision import models
from collections import namedtuple
from pathlib import Path
from ditk import logging


DEFAULT_TORCH_HOME = Path(__file__).resolve().parents[3] / "tokenizer_pretrained_vgg"
VGG16_CKPT_NAME = "vgg16-397923af.pth"


class LPIPS(nn.Module):
    # Learned perceptual metric
    def __init__(self, use_dropout: bool = True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]  # vg16 features
        self.torch_home = resolve_torch_home()

        self.net = vgg16(pretrained=True, requires_grad=False, torch_home=self.torch_home)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.load_from_pretrained()
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self) -> None:
        """
        Load LPIPS linear layer weights (vgg.pth) from TORCH_HOME directory.

        Raises:
            FileNotFoundError: If checkpoint file cannot be loaded.
            RuntimeError: If state dict loading fails.
        """
        try:
            logging.info(f"Loading LPIPS pretrained weights from TORCH_HOME: {self.torch_home}")
            ckpt = get_ckpt_path(name="vgg_lpips", root=self.torch_home)

            if not os.path.exists(ckpt):
                error_msg = f"Checkpoint file not found: {ckpt}"
                logging.error(error_msg)
                raise FileNotFoundError(error_msg)

            logging.info(f"Loading checkpoint from: {ckpt}")
            state_dict = torch.load(ckpt, map_location=torch.device("cpu"))
            self.load_state_dict(state_dict, strict=False)
            logging.info(f"Successfully loaded LPIPS pretrained weights from: {ckpt}")

        except FileNotFoundError as e:
            logging.error(f"Failed to load LPIPS checkpoint: {e}")
            raise
        except Exception as e:
            error_msg = f"Failed to load LPIPS pretrained weights: {e}"
            logging.error(error_msg)
            raise RuntimeError(error_msg) from e

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
    def __init__(self, requires_grad: bool = False, pretrained: bool = True, torch_home: Path = None) -> None:
        super(vgg16, self).__init__()
        logging.info("Loading vgg16 backbone...")
        vgg_model = models.vgg16(weights=None)
        if pretrained:
            ckpt = get_vgg16_ckpt_path(torch_home or resolve_torch_home())
            logging.info(f"Loading vgg16 backbone weights from: {ckpt}")
            state_dict = torch.load(ckpt, map_location=torch.device("cpu"))
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            vgg_model.load_state_dict(state_dict)
        vgg_pretrained_features = vgg_model.features
        logging.info("vgg16 backbone loaded.")
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
# *************** Utilities to load pretrained vgg locally ************
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


def md5_hash(path: str) -> str:
    with open(path, "rb") as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()


def resolve_torch_home() -> Path:
    return Path(os.environ.get("TORCH_HOME", DEFAULT_TORCH_HOME)).expanduser().resolve()


def get_ckpt_path(name: str, root: str, check: bool = False) -> str:
    assert name in URL_MAP
    path = os.path.join(str(root), CKPT_MAP[name])
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing LPIPS checkpoint: {path}. Set TORCH_HOME to a directory containing {CKPT_MAP[name]}."
        )
    if check and not md5_hash(path) == MD5_MAP[name]:
        raise RuntimeError(f"MD5 mismatch for {path}: expected {MD5_MAP[name]}, got {md5_hash(path)}")
    return path


def get_vgg16_ckpt_path(root: Path) -> Path:
    candidates = [
        root / "hub" / "checkpoints" / VGG16_CKPT_NAME,
        root / VGG16_CKPT_NAME,
    ]
    for path in candidates:
        if path.exists():
            return path
    candidate_text = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Missing VGG16 checkpoint. Expected one of: {candidate_text}. "
        f"Set TORCH_HOME to the pretrained VGG directory."
    )
