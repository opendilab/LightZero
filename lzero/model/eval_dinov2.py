import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO
import os # 导入 os 模块以检查文件是否存在

# -----------------------------------------------------------------------------
# 1. DinoV2Encoder 定义 (保持不变)
# -----------------------------------------------------------------------------
class DinoV2Encoder(nn.Module):
    def __init__(self, name="dinov2_vits14", feature_key="x_norm_clstoken"):
        super().__init__()
        self.name = name
        # 使用 torch.hub.set_dir() 来指定一个缓存目录，避免每次都下载
        # torch.hub.set_dir('/path/to/your/hub/cache') 
        self.base_model = torch.hub.load("facebookresearch/dinov2", name)
        self.feature_key = feature_key
        self.emb_dim = self.base_model.num_features
        if feature_key == "x_norm_patchtokens":
            self.latent_ndim = 2
        elif feature_key == "x_norm_clstoken":
            self.latent_ndim = 1
        else:
            raise ValueError(f"Invalid feature key: {feature_key}")

        self.patch_size = self.base_model.patch_size

    def forward(self, x):
        """
        输入 x 应该是已经预处理好的张量，shape: (B, 3, H, W)
        """
        emb = self.base_model.forward_features(x)[self.feature_key]
        if self.latent_ndim == 1:
            emb = emb.unsqueeze(1) # dummy patch dim
        return emb

# -----------------------------------------------------------------------------
# 2. 准备图像预处理函数 (已修复和优化)
# -----------------------------------------------------------------------------
def prepare_image(image_source: str):
    """
    从给定的来源 (URL或本地路径) 加载图片并进行预处理。
    支持 .jpg, .png, .webp 等 Pillow 支持的格式。
    """
    img = None
    try:
        # 判断输入是 URL 还是本地文件路径
        if image_source.startswith('http://') or image_source.startswith('https://'):
            # --- 处理 URL ---
            response = requests.get(image_source)
            response.raise_for_status()  # 如果下载失败则抛出异常
            img = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            # --- 处理本地文件路径 ---
            if not os.path.exists(image_source):
                raise FileNotFoundError(f"本地文件未找到: {image_source}")
            img = Image.open(image_source).convert("RGB")
            
        # DINOv2 推荐的预处理
        # Patch size 是 14，所以输入尺寸最好是 14 的倍数
        # 官方推荐尺寸是 518x518
        transform = transforms.Compose([
            transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(518),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        return transform(img).unsqueeze(0) # 增加 batch 维度
        
    except Exception as e:
        print(f"无法加载或处理图片 '{image_source}': {e}")
        return None

# -----------------------------------------------------------------------------
# 3. 编写测试主函数 (已更新变量名和结论)
# -----------------------------------------------------------------------------
def test_dinov2_pretrained_features():
    """
    测试 DinoV2Encoder 是否加载了预训练权重并能提取有意义的语义特征。
    """
    print("正在初始化 DinoV2Encoder...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    try:
        encoder = DinoV2Encoder(name="dinov2_vits14")
        encoder.to(device)
        encoder.eval() # 设置为评估模式
    except Exception as e:
        print(f"初始化 DinoV2Encoder 失败: {e}")
        print("请检查您的网络连接或 PyTorch Hub 配置。")
        return

    print("模型加载成功。准备加载并预处理图片...")

    # --- 图像来源 ---
    # 您现在可以混合使用本地路径和 URL
    cat_image_path_A = "/mnt/nfs/zhangjinouwen/puyuan/LightZero/lzero/model/cat_1.jpg"
    cat_image_path_B = "/mnt/nfs/zhangjinouwen/puyuan/LightZero/lzero/model/cat_2.webp"
    dog_image_path = "/mnt/nfs/zhangjinouwen/puyuan/LightZero/lzero/model/dog.webp"
    # 为了演示，我们再加一个来自网络的汽车图片
    car_image_url = "https://images.unsplash.com/photo-1503376780353-7e6692767b70"


    # 加载和预处理
    img_cat_A = prepare_image(cat_image_path_A)
    img_cat_B = prepare_image(cat_image_path_B)
    img_dog = prepare_image(dog_image_path)
    img_car = prepare_image(car_image_url)

    # 检查所有图片是否加载成功
    if any(img is None for img in [img_cat_A, img_cat_B, img_dog, img_car]):
        print("部分或全部图片加载失败，测试中止。")
        return
        
    img_cat_A, img_cat_B, img_dog, img_car = [img.to(device) for img in [img_cat_A, img_cat_B, img_dog, img_car]]

    print("图片处理完成，开始提取特征向量...")

    with torch.no_grad():
        vec_cat_A = encoder(img_cat_A).squeeze(1)
        vec_cat_B = encoder(img_cat_B).squeeze(1)
        vec_dog = encoder(img_dog).squeeze(1)
        vec_car = encoder(img_car).squeeze(1)

    print("特征提取完成，开始计算余弦相似度...")

    cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
    
    sim_cats = cos_sim(vec_cat_A, vec_cat_B)
    sim_cat_dog = cos_sim(vec_cat_A, vec_dog)
    sim_cat_car = cos_sim(vec_cat_A, vec_car)

    print("\n" + "="*50)
    print("测试结果:")
    print(f"  - 两张'猫'图片的特征相似度: {sim_cats.item():.4f}")
    print(f"  - '猫'和'狗'图片的特征相似度: {sim_cat_dog.item():.4f}")
    print(f"  - '猫'和'汽车'图片的特征相似度: {sim_cat_car.item():.4f}")
    print("="*50 + "\n")

    # 结论分析
    print("结论分析:")
    # 预期：猫-猫 相似度 > 猫-狗 相似度 > 猫-车 相似度
    all_passed = True
    if sim_cats.item() > sim_cat_dog.item():
        print("✅ 符合预期: '猫-猫' 相似度高于 '猫-狗'。")
    else:
        print("❌ 不符预期: '猫-猫' 相似度未高于 '猫-狗'。")
        all_passed = False
        
    if sim_cat_dog.item() > sim_cat_car.item():
        print("✅ 符合预期: '猫-狗' 相似度高于 '猫-汽车'。")
    else:
        print("❌ 不符预期: '猫-狗' 相似度未高于 '猫-汽车'。")
        all_passed = False

    if all_passed:
        print("\n🎉 测试通过！结果表明模型能够区分不同语义层级的相似度。")
        print("   模型理解了'猫'和'狗'同属'动物'类别，比'汽车'更相似，但不如两只'猫'之间相似。")
    else:
        print("\n⚠️ 测试部分失败！请检查模型或图片质量。")

if __name__ == '__main__':
    # 为了运行此脚本，请确保已安装 requests 和 Pillow
    # pip install requests Pillow
    test_dinov2_pretrained_features()