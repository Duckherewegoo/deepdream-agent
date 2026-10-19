"""PyTorch 版 DeepDream 核心实现（商业级性能重构版）
基于 InceptionV3 模型，优化梦境参数，体现真实梦境特征
"""
import json
import logging
import os
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
from torch import nn
from torchvision import models, transforms

from ..config.settings import DEBUG_MODE, DREAM_CONFIGS, DREAM_LAYERS

logger = logging.getLogger(__name__)


class FeatureExtractor(nn.Module):
    """
    极客版特征提取器：取代传统的 register_forward_hook。
    优势：
    1. 手动控制前向传播流程，拿到目标层后立刻早停（Early Stopping）。
    2. 避免 Hook 无法注销导致的显存泄漏。
    3. 减少重复遍历子模块的开销。
    """
    def __init__(self, model: nn.Module, target_layers: list[str]):
        super().__init__()
        self.model = model
        self.target_layers = set(target_layers)
        self.features = {}
        
        # 预计算要遍历的层
        self.children = list(self.model.named_children())
        self.max_layer_idx = -1
        for i, (name, _) in enumerate(self.children):
            if name in self.target_layers:
                self.max_layer_idx = max(self.max_layer_idx, i)
                
        if self.max_layer_idx == -1:
            raise ValueError(f"未在模型中找到任何目标层: {target_layers}")

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        self.features = {}
        
        # 按顺序执行，命中最后一层直接 break
        for i, (name, module) in enumerate(self.children):
            x = module(x)
            if name in self.target_layers:
                self.features[name] = x
                
            if i >= self.max_layer_idx:
                break
                
        return [self.features[name] for name in self.target_layers if name in self.features]


class DeepDreamGenerator:
    """DeepDream 生成器 - PyTorch 实现（商业级性能版）"""

    def __init__(self, device: str | None = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 1. 加载模型并冻结权重
        self.model = models.inception_v3(
            weights=models.Inception_V3_Weights.IMAGENET1K_V1,
            transform_input=False
        )
        self.model.eval()
        self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = False

        # 2. 预处理与反预处理
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.denormalize = transforms.Compose([
            transforms.Normalize(
                mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
            ),
        ])

        # 3. 特征提取器缓存池
        self._layer_extractors: dict[tuple, FeatureExtractor] = {}
        self.debug = DEBUG_MODE
        logger.info(f"[DREAM] 引擎初始化完成 | 设备: {self.device} | 调试: {self.debug}")

    @staticmethod
    def _align_size(h: int, w: int) -> tuple[int, int]:
        """强制对齐到 8 的倍数（Inception V3 下采样要求）"""
        return max(8, int(h // 8 * 8)), max(8, int(w // 8 * 8))

    def _get_layer_extractor(self, layer_names: list[str]) -> FeatureExtractor:
        """获取并缓存特征提取器"""
        cache_key = tuple(sorted(layer_names))
        if cache_key not in self._layer_extractors:
            extractor = FeatureExtractor(self.model, layer_names)
            extractor.to(self.device)
            extractor.eval()
            self._layer_extractors[cache_key] = extractor
        return self._layer_extractors[cache_key]

    def _load_image(self, image_path: str, cfg: dict) -> torch.Tensor:
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        
        max_size = cfg.get("MAX_SIZE", 800)
        max_side = max(h, w)
        if max_side > max_size:
            scale = max_size / max_side
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        with torch.no_grad():
            img_tensor = self.preprocess(img).unsqueeze(0)
            return img_tensor.to(self.device)

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        with torch.no_grad():
            tensor = self.denormalize(tensor.cpu())
            tensor = torch.clamp(tensor, 0, 1)
        return transforms.ToPILImage()(tensor)

    def _add_gradient_noise(self, grad: torch.Tensor, cfg: dict, dream_type: str) -> torch.Tensor:
        noise_val = cfg.get("GRADIENT_NOISE", 0.0)
        if noise_val <= 0:
            return grad

        # 使用 Tensor 运算，避免 .item() 导致 CPU-GPU 同步卡顿
        std_val = torch.std(grad)
        noise_scale = noise_val * std_val

        if dream_type == "mist":
            noise_scale = noise_scale * (1.0 - 0.3 * torch.sigmoid(noise_scale))
        elif dream_type == "crazy":
            noise_scale = noise_scale * (1.0 + 0.4 * torch.tanh(noise_scale))
        elif dream_type == "nightmare":
            noise_scale = noise_scale * (1.0 + 0.2 * torch.tanh(noise_scale))

        return grad + torch.randn_like(grad) * noise_scale

    def _compute_loss(self, features: list[torch.Tensor], cfg: dict, dream_type: str) -> torch.Tensor:
        loss = torch.tensor(0.0, device=self.device)
        scale = cfg.get("GRADIENT_SCALE", 1.0)

        for i, feat in enumerate(features):
            if dream_type == "nightmare":
                weight = 1.0 + i * 0.35
            elif dream_type == "crazy":
                weight = 1.2 + i * 0.25
            elif dream_type == "sweet":
                weight = 1.0 - i * 0.15
            elif dream_type == "mist":
                weight = 1.0 - i * 0.25
            else:
                weight = 1.0
            loss = loss + torch.mean(feat) * weight * scale

        return loss

    def _dream_step(self, image: torch.Tensor, extractor: FeatureExtractor, cfg: dict,
                    dream_type: str, octave_idx: int, iter_idx: int) -> torch.Tensor:

        progress = iter_idx / max(cfg.get("ITER_PER_OCTAVE", 80) - 1, 1)
        step_size = cfg.get("STEP_SIZE", 0.01)

        # 动态步长
        if cfg.get("DYNAMIC_STEP", False):
            if dream_type == "crazy":
                step_size *= (1 + 0.9 * (progress ** 2))
            elif dream_type == "nightmare":
                step_size *= (1 + 0.6 * progress)
            elif dream_type == "mist":
                step_size *= (1 + 0.15 * progress)
            else:
                step_size *= (1 + 0.3 * progress)

        img = image.detach().clone().requires_grad_(True)
        shift_x, shift_y = 0, 0

        # 随机偏移防伪影
        if cfg.get("SAFE_SHIFT", False):
            shift_x = np.random.randint(-3, 4)
            shift_y = np.random.randint(-3, 4)
            img_shifted = torch.roll(torch.roll(img, shift_x, dims=3), shift_y, dims=2)
        else:
            img_shifted = img

        features = extractor(img_shifted)
        loss = self._compute_loss(features, cfg, dream_type)

        # 计算梯度并立即释放计算图
        grad = torch.autograd.grad(loss, img_shifted, retain_graph=False, create_graph=False)[0]
        
        # 梯度数值保护
        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.01, neginf=-0.01)
        grad = torch.clamp(grad, min=-0.1, max=0.1)
        
        grad_std = torch.std(grad, dim=[2, 3], keepdim=True).clamp(min=1e-8)
        grad = grad / (grad_std + 1e-8)

        # 抵消偏移
        if cfg.get("SAFE_SHIFT", False):
            grad = torch.roll(torch.roll(grad, -shift_x, dims=3), -shift_y, dims=2)

        grad = self._add_gradient_noise(grad, cfg, dream_type)

        # 特殊梦境增强
        if "STAR_BRIGHTNESS" in cfg:
            brightness = float(cfg["STAR_BRIGHTNESS"])
            grad = grad * (1.0 + brightness * torch.abs(grad))

        if cfg.get("WAVE_FLOW", 0) > 0:
            wave_intensity = float(cfg["WAVE_FLOW"])
            _, _, h, w = grad.shape
            grid_y, _ = torch.meshgrid(
                torch.arange(h, device=self.device, dtype=torch.float32),
                torch.arange(w, device=self.device, dtype=torch.float32),
                indexing='ij'
            )
            x_shift = wave_intensity * torch.sin(grid_y / 10.0)
            grad = torch.roll(grad, shifts=int(x_shift.mean().item()), dims=3)

        img = img.detach() + grad * step_size

        # 现实融合
        fusion = float(cfg.get("REALITY_FUSION", 0))
        if fusion > 0:
            img = img * (1 - fusion) + image * fusion

        return torch.clamp(img, -3, 3)

    def _generate_base(self, image: torch.Tensor, extractor: FeatureExtractor,
                       cfg: dict, dream_type: str) -> torch.Tensor:
        _, _, h, w = image.shape

        octave_layers = cfg.get("OCTAVE_LAYERS", 3)
        octave_scale = cfg.get("OCTAVE_SCALE", 0.75)
        iter_per_octave = cfg.get("ITER_PER_OCTAVE", 80)

        shapes = [self._align_size(int(h * (octave_scale ** i)), int(w * (octave_scale ** i))) 
                  for i in range(octave_layers)][::-1]

        current_img = F.interpolate(image, size=shapes[0], mode='bilinear', align_corners=False)

        for octave_idx, size in enumerate(shapes):
            current_img = F.interpolate(current_img, size=size, mode='bilinear', align_corners=False)

            for i in range(iter_per_octave):
                current_img = self._dream_step(current_img, extractor, cfg, dream_type, octave_idx, i)

        orig_h, orig_w = self._align_size(h, w)
        return F.interpolate(current_img, size=(orig_h, orig_w), mode='bilinear', align_corners=False)

    # ==================== 公共接口 ====================
    def generate(self, input_path: str, dream_type: str = "normal",
                 output_dir: str = "./dream_outputs",
                 custom_cfg: dict | None = None,
                 custom_layers: list[str] | None = None) -> str:

        start_time = time.time()
        try:
            os.makedirs(output_dir, exist_ok=True)

            # 合并配置
            cfg = DREAM_CONFIGS.get(dream_type, DREAM_CONFIGS["normal"]).copy()
            if custom_cfg:
                cfg.update(custom_cfg)

            layer_names = custom_layers or DREAM_LAYERS.get(dream_type, ["Mixed_5b"])

            base_name = os.path.splitext(os.path.basename(input_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}_{dream_type}.png")
            meta_path = os.path.splitext(output_path)[0] + ".meta.json"

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            img_tensor = self._load_image(input_path, cfg)
            extractor = self._get_layer_extractor(layer_names)

            result_tensor = self._generate_base(img_tensor, extractor, cfg, dream_type)

            result_img = self._tensor_to_pil(result_tensor)
            result_img = self._post_process(result_img, dream_type)
            result_img.save(output_path, format="PNG", optimize=True)

            # 强制垃圾回收释放显存
            del img_tensor, result_tensor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 生成 Meta
            serializable_cfg = {k: v if isinstance(v, (int, float, str, bool, list, dict)) else str(v) 
                                for k, v in cfg.items()}
            meta_data = {
                "dream_type": dream_type,
                "input_image": os.path.basename(input_path),
                "output_image": os.path.basename(output_path),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "config": serializable_cfg,
                "layers_used": layer_names,
                "device": str(self.device),
                "generation_time_sec": round(time.time() - start_time, 2)
            }

            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta_data, f, ensure_ascii=False, indent=4)

            logger.info(f"[DREAM] 生成成功! 耗时: {time.time() - start_time:.2f}s | 输出: {output_path}")
            return output_path

        except Exception as e:
            logger.exception(f"[DREAM] 生成失败! 耗时: {time.time() - start_time:.2f}s | 错误: {e}")
            raise RuntimeError(f"DeepDream 生成失败: {e}") from e

    def _post_process(self, image: Image.Image, dream_type: str) -> Image.Image:
        if dream_type == "sweet":
            image = image.filter(ImageFilter.GaussianBlur(radius=0.8))
            image = ImageEnhance.Brightness(image).enhance(1.1)
            image = ImageEnhance.Color(image).enhance(1.05)
        elif dream_type == "nightmare":
            image = image.filter(ImageFilter.SHARPEN)
            image = ImageEnhance.Brightness(image).enhance(0.9)
            image = ImageEnhance.Contrast(image).enhance(1.1)
        elif dream_type == "crazy":
            image = ImageEnhance.Color(image).enhance(1.2)
        return image

    def clear_cache(self):
        """彻底释放显存与特征提取器"""
        self._layer_extractors.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect() # 额外清理多进程共享显存
