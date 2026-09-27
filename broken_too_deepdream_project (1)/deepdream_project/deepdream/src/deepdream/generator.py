"""PyTorch 版 DeepDream 核心实现
基于 InceptionV3 模型，保留原有5种梦境配置参数
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageFilter
from typing import Dict, List, Optional

from ..config.settings import DREAM_CONFIGS, DREAM_LAYERS


class DeepDreamGenerator:
    """DeepDream 生成器 - PyTorch 实现"""

    def __init__(self, device: Optional[str] = None):
        """初始化 DeepDream 生成器

        Args:
            device: 计算设备，自动检测 GPU/CPU
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 加载预训练 InceptionV3 模型
        self.model = models.inception_v3(pretrained=True, transform_input=False)
        self.model.eval()
        self.model.to(self.device)

        # 图像预处理（对应 InceptionV3 的预处理方式）
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 反归一化，用于将张量转回图像
        self.denormalize = transforms.Compose([
            transforms.Normalize(
                mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
            ),
        ])

        # 缓存各层的 feature extractor
        self._layer_extractors = {}

    def _get_layer_extractor(self, layer_names: List[str]) -> nn.Module:
        """获取指定层的特征提取器

        Args:
            layer_names: 要提取特征的层名称列表

        Returns:
            特征提取模型
        """
        cache_key = tuple(sorted(layer_names))
        if cache_key in self._layer_extractors:
            return self._layer_extractors[cache_key]

        # 创建特征提取模型
        class FeatureExtractor(nn.Module):
            def __init__(self, model, target_layers):
                super().__init__()
                self.model = model
                self.target_layers = target_layers
                self.features = {}

                # 注册前向钩子
                for name, module in self.model.named_modules():
                    if name in target_layers:
                        module.register_forward_hook(self._make_hook(name))

            def _make_hook(self, name):
                def hook(module, input, output):
                    self.features[name] = output
                return hook

            def forward(self, x):
                self.features = {}
                _ = self.model(x)
                return [self.features[name] for name in self.target_layers]

        extractor = FeatureExtractor(self.model, layer_names)
        extractor.eval()
        extractor.to(self.device)
        self._layer_extractors[cache_key] = extractor
        return extractor

    def _load_image(self, image_path: str, cfg: Dict) -> torch.Tensor:
        """加载并预处理图像

        Args:
            image_path: 图片路径
            cfg: 梦境配置

        Returns:
            预处理后的图像张量 [1, C, H, W]
        """
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        max_side = max(h, w)

        # 按最大尺寸缩放
        if max_side > cfg["MAX_SIZE"]:
            scale = cfg["MAX_SIZE"] / max_side
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)

        # 预处理并添加 batch 维度
        img_tensor = self.preprocess(img).unsqueeze(0)
        return img_tensor.to(self.device)

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """将张量转换为 PIL 图像

        Args:
            tensor: 图像张量 [1, C, H, W] 或 [C, H, W]

        Returns:
            PIL 图像
        """
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)

        # 反归一化
        tensor = self.denormalize(tensor.cpu())

        # 裁剪到 [0, 1] 范围
        tensor = torch.clamp(tensor, 0, 1)

        # 转为 PIL 图像
        img = transforms.ToPILImage()(tensor)
        return img

    def _dream_step(
        self,
        image: torch.Tensor,
        extractor: nn.Module,
        step_size: float,
        reality_fusion: float,
        origin: torch.Tensor,
        safe_shift: bool = False,
    ) -> torch.Tensor:
        """单步 DeepDream 迭代

        Args:
            image: 当前图像张量
            extractor: 特征提取器
            step_size: 步长
            reality_fusion: 现实融合比例
            origin: 原始图像（用于融合）
            safe_shift: 是否使用随机滚动避免边界伪影

        Returns:
            更新后的图像张量
        """
        image = image.detach().requires_grad_(True)

        # Safe Shift: 随机滚动图像避免边界伪影
        shift_x, shift_y = 0, 0
        if safe_shift:
            shift_x = np.random.randint(-2, 3)
            shift_y = np.random.randint(-2, 3)
            image = torch.roll(torch.roll(image, shift_x, dims=3), shift_y, dims=2)

        # 前向传播获取特征
        features = extractor(image)

        # 计算损失（各层特征图均值之和）
        loss = torch.tensor(0.0, device=self.device)
        for feat in features:
            loss = loss + torch.mean(feat)

        # 反向传播计算梯度
        loss.backward()

        # 归一化梯度
        grad = image.grad.data
        grad = grad / (torch.std(grad) + 1e-8)

        # 更新图像
        image = image.detach() + grad * step_size

        # 回滚 shift
        if safe_shift:
            image = torch.roll(torch.roll(image, -shift_x, dims=3), -shift_y, dims=2)

        # 现实融合（与原图混合，保留一定真实感）
        if reality_fusion > 0:
            image = image * (1 - reality_fusion) + origin * reality_fusion

        # 裁剪到有效范围
        image = torch.clamp(image, -3, 3)

        return image.detach()

    def _generate_base(
        self,
        image: torch.Tensor,
        extractor: nn.Module,
        cfg: Dict,
    ) -> torch.Tensor:
        """基础梦境生成（多尺度 octave）

        Args:
            image: 输入图像张量
            extractor: 特征提取器
            cfg: 梦境配置

        Returns:
            生成的梦境图像张量
        """
        _, _, h, w = image.shape
        original_shape = (h, w)

        # 计算各 octave 尺寸
        shapes = []
        for i in range(cfg["OCTAVE_LAYERS"]):
            scale = cfg["OCTAVE_SCALE"] ** i
            shapes.append((int(h * scale), int(w * scale)))
        shapes = shapes[::-1]  # 从小到大

        # 从最小尺度开始
        current_img = torch.nn.functional.interpolate(
            image, size=shapes[0], mode='bilinear', align_corners=False
        )

        for size in shapes:
            # 缩放到当前尺度
            current_img = torch.nn.functional.interpolate(
                current_img, size=size, mode='bilinear', align_corners=False
            )
            origin = torch.nn.functional.interpolate(
                image, size=size, mode='bilinear', align_corners=False
            )

            # 迭代
            for _ in range(cfg["ITER_PER_OCTAVE"]):
                current_img = self._dream_step(
                    current_img,
                    extractor,
                    cfg["STEP_SIZE"],
                    cfg["REALITY_FUSION"],
                    origin,
                    safe_shift=cfg.get("SAFE_SHIFT", False),
                )

        # 恢复原始尺寸
        current_img = torch.nn.functional.interpolate(
            current_img, size=original_shape, mode='bilinear', align_corners=False
        )

        return current_img

    def _generate_mist(
        self,
        image: torch.Tensor,
        extractor: nn.Module,
        cfg: Dict,
    ) -> torch.Tensor:
        """迷雾梦生成（带 safe_shift 的多尺度）

        Args:
            image: 输入图像张量
            extractor: 特征提取器
            cfg: 梦境配置

        Returns:
            生成的梦境图像张量
        """
        _, _, h, w = image.shape
        original_shape = (h, w)

        shapes = []
        for i in range(cfg["OCTAVE_LAYERS"]):
            scale = cfg["OCTAVE_SCALE"] ** i
            shapes.append((int(h * scale), int(w * scale)))
        shapes = shapes[::-1]

        current = torch.nn.functional.interpolate(
            image, size=shapes[0], mode='bilinear', align_corners=False
        )

        for size in shapes:
            current = torch.nn.functional.interpolate(
                current, size=size, mode='bilinear', align_corners=False
            )
            origin = torch.nn.functional.interpolate(
                image, size=size, mode='bilinear', align_corners=False
            )

            for _ in range(cfg["ITER_PER_OCTAVE"]):
                # 迷雾梦每次迭代都做 safe_shift
                if cfg["SAFE_SHIFT"]:
                    shift_x = np.random.randint(-2, 3)
                    shift_y = np.random.randint(-2, 3)
                    current = torch.roll(torch.roll(current, shift_x, dims=3), shift_y, dims=2)

                current = self._dream_step(
                    current,
                    extractor,
                    cfg["STEP_SIZE"],
                    cfg["REALITY_FUSION"],
                    origin,
                    safe_shift=False,  # 已经在外面做了 shift
                )

        current = torch.nn.functional.interpolate(
            current, size=original_shape, mode='bilinear', align_corners=False
        )

        return current

    def _generate_crazy(
        self,
        image: torch.Tensor,
        extractor: nn.Module,
        cfg: Dict,
    ) -> torch.Tensor:
        """疯狂梦生成（高强度、多层、多尺度）

        Args:
            image: 输入图像张量
            extractor: 特征提取器
            cfg: 梦境配置

        Returns:
            生成的梦境图像张量
        """
        _, _, h, w = image.shape
        original_shape = (h, w)

        shapes = []
        for i in range(cfg["OCTAVE_LAYERS"]):
            scale = cfg["OCTAVE_SCALE"] ** i
            shapes.append((int(h * scale), int(w * scale)))
        shapes = shapes[::-1]

        current = torch.nn.functional.interpolate(
            image, size=shapes[0], mode='bilinear', align_corners=False
        )

        for size in shapes:
            current = torch.nn.functional.interpolate(
                current, size=size, mode='bilinear', align_corners=False
            )

            for _ in range(cfg["ITER_PER_OCTAVE"]):
                # 疯狂梦：每次都随机滚动，不做 reality fusion
                shift_x = np.random.randint(-2, 3)
                shift_y = np.random.randint(-2, 3)
                current = torch.roll(torch.roll(current, shift_x, dims=3), shift_y, dims=2)

                current = current.detach().requires_grad_(True)
                features = extractor(current)

                loss = torch.tensor(0.0, device=self.device)
                for feat in features:
                    loss = loss + torch.mean(feat)

                loss.backward()
                grad = current.grad.data
                grad = grad / (torch.std(grad) + 1e-8)

                current = current.detach() + grad * cfg["STEP_SIZE"]
                current = torch.clamp(current, -3, 3)

        current = torch.nn.functional.interpolate(
            current, size=original_shape, mode='bilinear', align_corners=False
        )

        return current

    def _post_process(self, image: Image.Image, dream_type: str) -> Image.Image:
        """后处理：根据梦境类型添加滤镜效果

        Args:
            image: 生成的 PIL 图像
            dream_type: 梦境类型

        Returns:
            处理后的 PIL 图像
        """
        if dream_type == "sweet":
            # 美梦：高斯模糊增加柔和感
            image = image.filter(ImageFilter.GaussianBlur(radius=1.2))
        elif dream_type == "nightmare":
            # 噩梦：锐化增加恐怖感
            image = image.filter(ImageFilter.SHARPEN)

        return image

    def generate(self, input_path: str, dream_type: str, output_dir: str = "./dream_outputs") -> str:
        """生成梦境图片

        Args:
            input_path: 输入图片路径
            dream_type: 梦境类型 (normal/sweet/nightmare/mist/crazy)
            output_dir: 输出目录

        Returns:
            输出图片路径
        """
        # 验证梦境类型
        if dream_type not in DREAM_CONFIGS:
            dream_type = "normal"

        cfg = DREAM_CONFIGS[dream_type]
        layer_names = DREAM_LAYERS.get(dream_type, ["Mixed_5b"])

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 生成输出文件名
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_{dream_type}.png")

        try:
            # 加载图像
            img_tensor = self._load_image(input_path, cfg)

            # 获取特征提取器
            extractor = self._get_layer_extractor(layer_names)

            # 根据类型选择生成算法
            if dream_type in ["normal", "sweet", "nightmare"]:
                result_tensor = self._generate_base(img_tensor, extractor, cfg)
            elif dream_type == "mist":
                result_tensor = self._generate_mist(img_tensor, extractor, cfg)
            else:  # crazy
                result_tensor = self._generate_crazy(img_tensor, extractor, cfg)

            # 转为 PIL 图像
            result_img = self._tensor_to_pil(result_tensor)

            # 后处理
            result_img = self._post_process(result_img, dream_type)

            # 保存
            result_img.save(output_path)
            return output_path

        except Exception as e:
            # 出错时返回原图
            try:
                img = Image.open(input_path).convert("RGB")
                img.save(output_path)
            except Exception:
                pass
            return output_path
