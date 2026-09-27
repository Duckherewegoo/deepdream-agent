"""PyTorch 版 DeepDream 核心实现（日志增强 · 防呆加固版）
基于 InceptionV3 模型，优化梦境参数，体现真实梦境特征
"""
import os
import time
import traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image, ImageFilter, ImageEnhance
from typing import Dict, List, Optional
import json

from ..config.settings import DREAM_CONFIGS, DREAM_LAYERS, DEBUG_MODE


# ANSI 颜色（终端好看）
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
CYAN = '\033[96m'
RESET = '\033[0m'


class DeepDreamGenerator:
    """DeepDream 生成器 - PyTorch 实现（优化版）"""

    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
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
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
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
        print(
            f"{GREEN}[DREAM]{RESET} 生成器初始化完成，设备: {self.device}，调试模式: {self.debug}")

    # ==================== 工具方法 ====================
    @staticmethod
    def _align_size(h: int, w: int) -> tuple[int, int]:
        """✅ 强制对齐到 8 的倍数（Inception V3 硬性要求）"""
        h = max(8, int(h // 8 * 8))
        w = max(8, int(w // 8 * 8))
        return h, w

    def _get_layer_extractor(self, layer_names: List[str]) -> nn.Module:
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
                missing = [
                    n for n in self.target_layers if n not in self.features]
                if missing:
                    raise KeyError(f"特征层未找到: {missing}")
                return [self.features[name] for name in self.target_layers]

        extractor = FeatureExtractor(self.model, layer_names)
        extractor.eval()
        extractor.to(self.device)
        self._layer_extractors[cache_key] = extractor
        return extractor

    def _load_image(self, image_path: str, cfg: Dict) -> torch.Tensor:
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        print(f"{CYAN}[DREAM]{RESET} 原始图像尺寸: {w}x{h}")

        max_side = max(h, w)
        if max_side > cfg["MAX_SIZE"]:
            scale = cfg["MAX_SIZE"] / max_side
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)

        img_tensor = self.preprocess(img).unsqueeze(0)
        print(
            f"{CYAN}[DREAM]{RESET} 缩放后输入尺寸: {img_tensor.shape[3]}x{img_tensor.shape[2]}")
        return img_tensor.to(self.device)

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        tensor = self.denormalize(tensor.cpu())
        tensor = torch.clamp(tensor, 0, 1)
        return transforms.ToPILImage()(tensor)

    def _add_gradient_noise(self, grad: torch.Tensor, cfg: Dict, dream_type: str) -> torch.Tensor:
        if "GRADIENT_NOISE" not in cfg or cfg["GRADIENT_NOISE"] == 0:
            return grad

        # ✅ 只用标准差（标量），不用整张梯度图参与运算
        noise_scale = cfg["GRADIENT_NOISE"] * torch.std(grad)

        if dream_type == "mist":
            # 迷雾：均匀低频抖动
            noise_scale *= (1.0 - 0.3 * torch.sigmoid(noise_scale))
        elif dream_type == "crazy":
            # 疯狂：方向性增强
            noise_scale *= (1.0 + 0.4 * torch.tanh(noise_scale))
        elif dream_type == "nightmare":
            # 噩梦：高频刺耳（关键修复：用 noise_scale 而不是 grad）
            noise_scale *= (1.0 + 0.2 * torch.tanh(noise_scale))

        if self.debug:
            print(f"  {YELLOW}[NOISE]{RESET} 噪声强度: {noise_scale.item():.6f}")

        # ✅ 噪声本身是逐像素的，但缩放因子是标量，完美广播
        return grad + torch.randn_like(grad) * noise_scale
    # ==================== 核心逻辑 ====================

    def _compute_loss(self, features: List[torch.Tensor], cfg: Dict, dream_type: str) -> torch.Tensor:
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

    def _dream_step(self, image: torch.Tensor, extractor: nn.Module, cfg: Dict,
                    dream_type: str, octave_idx: int, iter_idx: int) -> torch.Tensor:

        # ✅ 动态步长
        progress = iter_idx / max(cfg["ITER_PER_OCTAVE"] - 1, 1)
        step_size = cfg["STEP_SIZE"]

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
            print(f"  {CYAN}[STEP]{RESET} Octave {octave_idx+1} | "
                  f"Iter {iter_idx+1} | 步长: {step_size:.4f}")

        # ✅ 强制开启梯度
        img = image.detach().clone().requires_grad_(True)
        shift_x, shift_y = 0, 0

        if cfg.get("SAFE_SHIFT", False):
            shift_x = np.random.randint(-3, 4)
            shift_y = np.random.randint(-3, 4)
            img_shifted = torch.roll(torch.roll(
                img, shift_x, dims=3), shift_y, dims=2)
        else:
            img_shifted = img

        features = extractor(img_shifted)
        loss = self._compute_loss(features, cfg, dream_type)

        if self.debug and iter_idx % 10 == 0:
            print(f"  {CYAN}[LOSS]{RESET} 损失值: {loss.item():.6f}")

        grad = torch.autograd.grad(
            loss, img_shifted, retain_graph=False, create_graph=False)[0]

        # ✅ 数值保护（噩梦模式保命符）
        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.01, neginf=-0.01)
        grad = torch.clamp(grad, min=-0.1, max=0.1)

        # ✅ 关键修复：按通道计算 std，保持维度 [1, 3, 1, 1]
        grad_std = torch.std(grad, dim=[2, 3], keepdim=True).clamp(min=1e-8)
        grad = grad / (grad_std + 1e-8)

        if self.debug and iter_idx % 10 == 0:
            print(f"  {CYAN}[GRAD]{RESET} 梯度标准差: {grad_std.mean().item():.6f}")

        if cfg.get("SAFE_SHIFT", False):
            grad = torch.roll(torch.roll(
                grad, -shift_x, dims=3), -shift_y, dims=2)

        grad = self._add_gradient_noise(grad, cfg, dream_type)

        img = img.detach() + grad * step_size

        fusion = float(cfg.get("REALITY_FUSION", 0))
        if fusion > 0:
            img = img * (1 - fusion) + image * fusion

        return torch.clamp(img, -3, 3)

    def _generate_base(self, image: torch.Tensor, extractor: nn.Module,
                       cfg: Dict, dream_type: str) -> torch.Tensor:
        _, _, h, w = image.shape

        # ✅ 计算 octave 尺寸（从大到小）
        shapes = []
        for i in range(cfg["OCTAVE_LAYERS"]):
            scale = cfg["OCTAVE_SCALE"] ** i
            shapes.append((int(h * scale), int(w * scale)))
        shapes = shapes[::-1]  # 从小到大

        print(f"\n{GREEN}[DREAM]{RESET} ▶️ 启动{dream_type}生成流水线")
        print(
            f"{GREEN}[DREAM]{RESET} Octave数量: {len(shapes)} | 单Octave迭代: {cfg['ITER_PER_OCTAVE']}")
        print(
            f"{GREEN}[DREAM]{RESET} 动态步长: {cfg.get('DYNAMIC_STEP', False)} | 现实融合: {cfg.get('REALITY_FUSION', 0)}")

        # 从最小尺度开始
        aligned_h, aligned_w = self._align_size(*shapes[0])
        current_img = F.interpolate(
            image, size=(aligned_h, aligned_w), mode='bilinear', align_corners=False)

        for octave_idx, size in enumerate(shapes):
            # ✅ 强制对齐尺寸
            aligned_h, aligned_w = self._align_size(size[0], size[1])
            scale_ratio = cfg["OCTAVE_SCALE"] ** (len(shapes) - octave_idx - 1)

            print(
                f"\n{GREEN}[DREAM]{RESET} ▶️ Octave {octave_idx+1}/{len(shapes)}")
            print(
                f"{GREEN}[DREAM]{RESET} 目标尺寸: {aligned_w}x{aligned_h} | 缩放比例: {scale_ratio:.3f}")

            current_img = F.interpolate(
                current_img, size=(aligned_h, aligned_w), mode='bilinear', align_corners=False)
            origin = F.interpolate(
                image, size=(aligned_h, aligned_w), mode='bilinear', align_corners=False)

            for i in range(cfg["ITER_PER_OCTAVE"]):
                current_img = self._dream_step(
                    current_img, extractor, cfg, dream_type, octave_idx, i
                )
                if (i + 1) % 20 == 0 or i == 0:
                    pct = ((i + 1) / cfg["ITER_PER_OCTAVE"]) * 100
                    print(
                        f"  {GREEN}[DREAM]{RESET} 迭代进度: {i+1}/{cfg['ITER_PER_OCTAVE']} ({pct:.1f}%)")

        # 恢复原始尺寸
        orig_h, orig_w = self._align_size(h, w)
        current_img = F.interpolate(
            current_img, size=(orig_h, orig_w), mode='bilinear', align_corners=False)

        return current_img

    # ==================== 公共接口 ====================
    def generate(self, input_path: str, dream_type: str = "normal",
                 output_dir: str = "./dream_outputs",
                 custom_cfg: Optional[Dict] = None,
                 custom_layers: Optional[List[str]] = None) -> str:

        start_time = time.time()
        try:
            print(f"\n{'='*60}")
            print(f"{GREEN}[DREAM]{RESET} 🚀 启动{dream_type}生成任务")
            print(f"{GREEN}[DREAM]{RESET} 输入路径: {input_path}")
            print(f"{'='*60}")

            os.makedirs(output_dir, exist_ok=True)

            cfg = custom_cfg if custom_cfg else DREAM_CONFIGS.get(
                dream_type, DREAM_CONFIGS["normal"])
            layer_names = custom_layers if custom_layers else DREAM_LAYERS.get(dream_type, [
                                                                               "Mixed_5b"])

            base_name = os.path.splitext(os.path.basename(input_path))[0]
            output_path = os.path.join(
                output_dir, f"{base_name}_{dream_type}.png")
            meta_path = os.path.splitext(output_path)[0] + ".meta.json"

            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            img_tensor = self._load_image(input_path, cfg)
            extractor = self._get_layer_extractor(layer_names)

            result_tensor = self._generate_base(
                img_tensor, extractor, cfg, dream_type)

            result_img = self._tensor_to_pil(result_tensor)
            result_img = self._post_process(result_img, dream_type, cfg)
            result_img.save(output_path, quality=95)

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
            print(f"\n{GREEN}[DREAM]{RESET} ✅ 生成成功！总耗时: {elapsed:.2f}s")
            print(f"{GREEN}[DREAM]{RESET} 输出图像: {output_path}")
            print(f"{GREEN}[DREAM]{RESET} Meta文件: {meta_path}")

            return output_path

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"\n{RED}[DREAM]{RESET} ❌ 生成失败！耗时: {elapsed:.2f}s")
            print(f"{RED}[DREAM]{RESET} 错误类型: {type(e).__name__}")
            print(f"{RED}[DREAM]{RESET} 错误详情: {str(e)}")
            traceback.print_exc()

            # ✅ 不再 fallback，直接抛异常
            raise RuntimeError(f"DeepDream 生成失败: {e}")

    def _post_process(self, image: Image.Image, dream_type: str, cfg: Dict) -> Image.Image:
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
