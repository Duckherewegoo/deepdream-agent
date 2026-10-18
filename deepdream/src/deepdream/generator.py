"""PyTorch 版 DeepDream 核心实现（生产级重构版）
基于 InceptionV3 模型，优化梦境参数，体现真实梦境特征
"""
import json
import logging
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
from torch import nn
from torchvision import models, transforms

from ..config.settings import DEBUG_MODE, DREAM_CONFIGS, DREAM_LAYERS

logger = logging.getLogger(__name__)


class DeepDreamGenerator:
    """DeepDream 生成器 - PyTorch 实现（生产最终版）"""

    def __init__(self, device: str | None = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = models.inception_v3(
            weights=models.Inception_V3_Weights.IMAGENET1K_V1,
            transform_input=False
        )
        self.model.eval()
        self.model.to(self.device)

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

        self._layer_extractors = {}
        for param in self.model.parameters():
            param.requires_grad = False

        self.debug = DEBUG_MODE
        logger.info(f"[DREAM] 生成器初始化完成 | 设备: {self.device} | 调试模式: {self.debug}")

    # ==================== 工具方法 ====================
    @staticmethod
    def _align_size(h: int, w: int) -> tuple[int, int]:
        """强制对齐到 8 的倍数（Inception V3 硬性要求）"""
        h = max(8, int(h // 8 * 8))
        w = max(8, int(w // 8 * 8))
        return h, w

    def _get_layer_extractor(self, layer_names: list[str]) -> nn.Module:
        cache_key = tuple(sorted(layer_names))
        if cache_key in self._layer_extractors:
            return self._layer_extractors[cache_key]

        class FeatureExtractor(nn.Module):
            def __init__(self, model, target_layers):
                super().__init__()
                self.model = model
                self.target_layers = target_layers
                self.features = {}

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
                missing = [n for n in self.target_layers if n not in self.features]
                if missing:
                    raise KeyError(f"特征层未找到: {missing}")
                return [self.features[name] for name in self.target_layers]

        extractor = FeatureExtractor(self.model, layer_names)
        extractor.eval()
        extractor.to(self.device)
        self._layer_extractors[cache_key] = extractor
        return extractor

    def _load_image(self, image_path: str, cfg: dict) -> torch.Tensor:
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        logger.debug(f"[DREAM] 原始图像尺寸: {w}x{h}")

        # 防御性获取 MAX_SIZE
        max_size = cfg.get("MAX_SIZE", 800)
        max_side = max(h, w)
        if max_side > max_size:
            scale = max_size / max_side
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)

        img_tensor = self.preprocess(img).unsqueeze(0)
        logger.debug(f"[DREAM] 缩放后输入尺寸: {img_tensor.shape[3]}x{img_tensor.shape[2]}")
        return img_tensor.to(self.device)

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        tensor = self.denormalize(tensor.cpu())
        tensor = torch.clamp(tensor, 0, 1)
        return transforms.ToPILImage()(tensor)

    def _add_gradient_noise(self, grad: torch.Tensor, cfg: dict, dream_type: str) -> torch.Tensor:
        noise_val = cfg.get("GRADIENT_NOISE", 0.0)
        if noise_val == 0:
            return grad

        std_val = torch.std(grad).item()
        noise_scale = noise_val * std_val

        if dream_type == "mist":
            noise_scale *= (1.0 - 0.3 * torch.sigmoid(torch.tensor(noise_scale)).item())
        elif dream_type == "crazy":
            noise_scale *= (1.0 + 0.4 * torch.tanh(torch.tensor(noise_scale)).item())
        elif dream_type == "nightmare":
            noise_scale *= (1.0 + 0.2 * torch.tanh(torch.tensor(noise_scale)).item())

        if self.debug:
            logger.debug(f"[NOISE] 噪声强度: {noise_scale:.6f}")

        return grad + torch.randn_like(grad) * noise_scale

    # ==================== 核心逻辑 ====================

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

    def _dream_step(self, image: torch.Tensor, extractor: nn.Module, cfg: dict,
                    dream_type: str, octave_idx: int, iter_idx: int) -> torch.Tensor:

        progress = iter_idx / max(cfg.get("ITER_PER_OCTAVE", 80) - 1, 1)
        step_size = cfg.get("STEP_SIZE", 0.01)

        if cfg.get("DYNAMIC_STEP", False):
            if dream_type == "crazy":
                step_size *= (1 + 0.9 * (progress ** 2))
            elif dream_type == "nightmare":
                step_size *= (1 + 0.6 * progress)
            elif dream_type == "mist":
                step_size *= (1 + 0.15 * progress)
            else:
                step_size *= (1 + 0.3 * progress)

        if self.debug and iter_idx % 10 == 0:
            logger.debug(f"[STEP] Octave {octave_idx+1} | Iter {iter_idx+1} | 步长: {step_size:.4f}")

        img = image.detach().clone().requires_grad_(True)
        shift_x, shift_y = 0, 0

        if cfg.get("SAFE_SHIFT", False):
            shift_x = np.random.randint(-3, 4)
            shift_y = np.random.randint(-3, 4)
            img_shifted = torch.roll(torch.roll(img, shift_x, dims=3), shift_y, dims=2)
        else:
            img_shifted = img

        features = extractor(img_shifted)
        loss = self._compute_loss(features, cfg, dream_type)

        if self.debug and iter_idx % 10 == 0:
            logger.debug(f"[LOSS] 损失值: {loss.item():.6f}")

        grad = torch.autograd.grad(loss, img_shifted, retain_graph=False, create_graph=False)[0]

        # 数值保护
        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.01, neginf=-0.01)
        grad = torch.clamp(grad, min=-0.1, max=0.1)

        # 按通道计算 std，保持维度 [1, 3, 1, 1]
        grad_std = torch.std(grad, dim=[2, 3], keepdim=True).clamp(min=1e-8)
        grad = grad / (grad_std + 1e-8)

        if self.debug and iter_idx % 10 == 0:
            logger.debug(f"[GRAD] 梯度标准差: {grad_std.mean().item():.6f}")

        if cfg.get("SAFE_SHIFT", False):
            grad = torch.roll(torch.roll(grad, -shift_x, dims=3), -shift_y, dims=2)

        grad = self._add_gradient_noise(grad, cfg, dream_type)

        # ================= 新梦境特性支持 =================
        # 1. 星空梦：星光闪烁增强
        if "STAR_BRIGHTNESS" in cfg:
            brightness = float(cfg["STAR_BRIGHTNESS"])
            grad = grad * (1.0 + brightness * torch.abs(grad))

        # 2. 海洋梦：波浪流场扭曲
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
        # ===================================================

        img = img.detach() + grad * step_size

        fusion = float(cfg.get("REALITY_FUSION", 0))
        if fusion > 0:
            img = img * (1 - fusion) + image * fusion

        return torch.clamp(img, -3, 3)

    def _generate_base(self, image: torch.Tensor, extractor: nn.Module,
                       cfg: dict, dream_type: str) -> torch.Tensor:
        _, _, h, w = image.shape

        shapes = []
        octave_layers = cfg.get("OCTAVE_LAYERS", 3)
        octave_scale = cfg.get("OCTAVE_SCALE", 0.75)
        iter_per_octave = cfg.get("ITER_PER_OCTAVE", 80)

        for i in range(octave_layers):
            scale = octave_scale ** i
            shapes.append((int(h * scale), int(w * scale)))
        shapes = shapes[::-1]

        logger.info(f"启动 {dream_type} 生成流水线 | Octaves: {len(shapes)} | Iters/Octave: {iter_per_octave}")

        aligned_h, aligned_w = self._align_size(*shapes[0])
        current_img = F.interpolate(image, size=(aligned_h, aligned_w), mode='bilinear', align_corners=False)

        for octave_idx, size in enumerate(shapes):
            aligned_h, aligned_w = self._align_size(size[0], size[1])
            scale_ratio = octave_scale ** (len(shapes) - octave_idx - 1)

            logger.info(f"Octave {octave_idx+1}/{len(shapes)} | 目标尺寸: {aligned_w}x{aligned_h} | 缩放比例: {scale_ratio:.3f}")

            current_img = F.interpolate(current_img, size=(aligned_h, aligned_w), mode='bilinear', align_corners=False)

            # 修复：删除了无用的 origin 计算

            for i in range(iter_per_octave):
                current_img = self._dream_step(current_img, extractor, cfg, dream_type, octave_idx, i)
                if (i + 1) % 20 == 0 or i == 0:
                    pct = ((i + 1) / iter_per_octave) * 100
                    logger.debug(f"迭代进度: {i+1}/{iter_per_octave} ({pct:.1f}%)")

        orig_h, orig_w = self._align_size(h, w)
        current_img = F.interpolate(current_img, size=(orig_h, orig_w), mode='bilinear', align_corners=False)

        return current_img

    # ==================== 公共接口 ====================
    def generate(self, input_path: str, dream_type: str = "normal",
                 output_dir: str = "./dream_outputs",
                 custom_cfg: dict | None = None,
                 custom_layers: list[str] | None = None) -> str:

        start_time = time.time()
        try:
            logger.info(f"启动 {dream_type} 生成任务 | 输入: {input_path}")

            os.makedirs(output_dir, exist_ok=True)

            # 核心修复：合并默认配置，防止 custom_cfg 缺少参数引发 KeyError
            default_cfg = DREAM_CONFIGS.get(dream_type, DREAM_CONFIGS["normal"]).copy()
            if custom_cfg:
                default_cfg.update(custom_cfg)
            cfg = default_cfg

            layer_names = custom_layers if custom_layers else DREAM_LAYERS.get(dream_type, ["Mixed_5b"])

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

            serializable_cfg = {}
            for k, v in cfg.items():
                if isinstance(v, (int, float, str, bool, list, dict)):
                    serializable_cfg[k] = v
                else:
                    serializable_cfg[k] = str(v)

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

            elapsed = time.time() - start_time
            logger.info(f"生成成功！总耗时: {elapsed:.2f}s | 输出: {output_path}")

            return output_path

        except Exception as e:
            elapsed = time.time() - start_time
            logger.exception(f"生成失败！耗时: {elapsed:.2f}s | 错误: {e!s}")
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
        self._layer_extractors.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
