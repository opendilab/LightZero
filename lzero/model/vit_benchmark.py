# benchmark_vit.py
import argparse, time, contextlib, importlib, random
from pathlib import Path

import torch, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

# ------------------------------------------------------------
# 1. 命令行
# ------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--device", default="cuda")
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--dataset", choices=["cifar", "fake"], default="cifar")
    p.add_argument("--fake_samples", type=int, default=10_000)
    p.add_argument("--num_classes", type=int, default=768)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--speed_rep", type=int, default=50)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--speed_only", action="store_true")
    return p.parse_args()

# ------------------------------------------------------------
# 2. 模型
# ------------------------------------------------------------
def build_models(device, img_size, num_classes):
    vit = importlib.import_module("vit")
    vite = importlib.import_module("vit_efficient")
    cfg = vite.ViTConfig(img_size=(img_size, img_size), num_classes=num_classes)

    baseline = vit.ViT(
        image_size=img_size, patch_size=8, num_classes=num_classes,
        dim=cfg.dim, depth=cfg.depth, heads=cfg.heads,
        mlp_dim=int(cfg.dim*cfg.mlp_ratio),
        dropout=cfg.dropout, emb_dropout=cfg.emb_dropout,
        final_norm_option_in_encoder="LayerNorm"
    ).to(device).eval()

    efficient = vite.VisionTransformer(cfg, final_norm="LayerNorm").to(device).eval()
    return baseline, efficient

def load_ckpt(model, path):
    if not path: return
    sd = torch.load(path, map_location="cpu")
    sd = sd.get("state_dict", sd)
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"[{model.__class__.__name__}] missing={len(miss)}  unexpected={len(unexp)}")

# ------------------------------------------------------------
# 3. 数据集
# ------------------------------------------------------------
class FakeSet(Dataset):
    """确定性伪造数据集：idx 与全局种子决定内容 -> 每次运行&每个模型都一致"""
    def __init__(self, n, img, classes, seed=123):
        self.n, self.img, self.classes, self.seed = n, img, classes, seed
    def __len__(self): return self.n
    def __getitem__(self, idx):
        g = torch.Generator().manual_seed(self.seed + idx)
        x = torch.randn(3, self.img, self.img, generator=g)
        y = torch.randint(0, self.classes, (1,), generator=g).item()
        return x, y

def get_loader(args):
    if args.dataset == "cifar":
        tf = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor()
        ])
        ds = datasets.CIFAR10("data", train=False, download=True, transform=tf)
    else:
        ds = FakeSet(args.fake_samples, args.img_size, args.num_classes)
    return DataLoader(ds, args.bs, shuffle=False,
                      num_workers=args.workers, pin_memory=True)

# ------------------------------------------------------------
# 4. 评估 / 对齐
# ------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, amp=False):
    acc = tot = 0
    ctx = torch.cuda.amp.autocast() if amp else contextlib.nullcontext()
    for x, y in loader:
        x, y = x.to(model.device), y.to(model.device)
        with ctx: out = model(x)
        acc += (out.argmax(-1) == y).sum().item()
        tot += y.numel()
    return acc / tot

@torch.no_grad()
def alignment(m1, m2, loader, amp=False, batches=20):
    cos = mse = 0.0
    ctx = torch.cuda.amp.autocast() if amp else contextlib.nullcontext()
    for idx, (x, _) in enumerate(loader):
        x = x.to(m1.device)
        with ctx:
            a, b = m1(x), m2(x)
        cos += F.cosine_similarity(a, b, dim=-1).mean().item()
        mse += F.mse_loss(a, b).item()
        if idx + 1 == batches: break
    return cos / batches, mse / batches

# ------------------------------------------------------------
# 5. 速度
# ------------------------------------------------------------
@torch.no_grad()
def benchmark(model, device, bs, img, rep, amp=False):
    x = torch.randn(bs, 3, img, img, device=device)
    ctx = torch.cuda.amp.autocast() if amp else contextlib.nullcontext()

    for _ in range(10):       # warm-up
        with ctx: model(x)
    if device.startswith("cuda"): torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(rep):
        with ctx: model(x)
    if device.startswith("cuda"): torch.cuda.synchronize()

    dt = (time.time() - t0) / rep
    return dt * 1000, bs / dt   # ms/img, imgs/s

# ------------------------------------------------------------
# 6. main
# ------------------------------------------------------------
def main():
    args = get_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    torch.manual_seed(42)
    random.seed(42)

    baseline, efficient = build_models(device, args.img_size, args.num_classes)
    baseline.device = efficient.device = device  # 方便 evaluate/alignment
    if args.ckpt:
        load_ckpt(baseline,  args.ckpt)
        load_ckpt(efficient, args.ckpt)

    # -------- 精度 & 对齐 --------
    if not args.speed_only:
        loader = get_loader(args)
        acc_b = evaluate(baseline, loader, args.amp)
        acc_e = evaluate(efficient, loader, args.amp)
        cos, mse = alignment(baseline, efficient, loader, args.amp)
        print("\n=== Accuracy ===")
        print(f"baseline : {acc_b*100:.2f}%")
        print(f"efficient: {acc_e*100:.2f}%")
        print("\n=== Alignment (first 20 batches) ===")
        print(f"cosine={cos:.6f} | mse={mse:.6e}")

    # -------- 速度 --------
    lat_b, thr_b = benchmark(baseline,  device, args.bs, args.img_size, args.speed_rep, args.amp)
    lat_e, thr_e = benchmark(efficient, device, args.bs, args.img_size, args.speed_rep, args.amp)
    print(f"\n=== Speed (bs={args.bs}, {'fp16' if args.amp else 'fp32'}) ===")
    print(f"baseline : {lat_b:6.2f} ms/img | {thr_b:,.1f} img/s")
    print(f"efficient: {lat_e:6.2f} ms/img | {thr_e:,.1f} img/s")
    print(f"Speed-up : {lat_b/lat_e:5.2f} ×   ({thr_e/thr_b:5.2f} × throughput)")

if __name__ == "__main__":
    main()

"""
# 1. 真实数据集 (CIFAR-10)
python benchmark_vit.py --device cuda --bs 64

# 2. 伪造数据集，仍然对齐检查 + 速度
python lzero/model/vit_benchmark.py --dataset fake --fake_samples 5000 --bs 256
"""

"""
=== Accuracy ===
baseline : 0.12%
efficient: 0.10%

=== Alignment (first 20 batches) ===
cosine=0.005367 | mse=1.989232e+00

=== Speed (bs=128, fp32) ===
baseline :  90.29 ms/img | 1,417.7 img/s
efficient:  86.11 ms/img | 1,486.4 img/s
Speed-up :  1.05 ×   ( 1.05 × throughput)
"""