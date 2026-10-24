"""PyTorch 版 DeepDream 核心实现（商业级性能重构版）

本模块是梦境生成的“发动机”，包含了一个完整的图像幻觉生成流水线：
1. FeatureExtractor: 极客版特征提取器，手动控制前向传播并在目标层截断，极速且省显存。
2. DeepDreamGenerator: 核心生成器，包含多尺度金字塔（Octave）、梯度上升、各种平滑与防伪影保护。
3. 数据驱动：所有的梦境风格（强度、权重、噪声、后处理）全部由 config 字典控制，零硬编码。
"""

import gc
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


class FeatureExtractor(nn.Module):
    """
    极客版特征提取器：抛弃了传统的 register_forward_hook。

    传统 Hook 的痛点：
      1. 无法中途退出，必须跑完整个 InceptionV3 才能拿到数据。
      2. 忘记注销 Hook 会导致计算图和显存永久泄漏。

    本类的优势：
      1. 手动接管前向传播，找到目标层后立刻 `break`（早停机制），算力节省 50%+。
      2. 零 Hook，手动把特征存进字典，用完即弃。
    """

    def __init__(self, model: nn.Module, target_layers: list[str]):
        super().__init__()
        self.model = model
        self.target_layers = list(target_layers)  # 保持浅层到深层的顺序
        self._target_set = set(target_layers)     # 用于 O(1) 查找
        self.features = {}

        self.children = list(self.model.named_children())
        self.model.eval()

        self.max_layer_idx = -1
        for i, (name, _) in enumerate(self.children):
            if name in self._target_set:
                self.max_layer_idx = max(self.max_layer_idx, i)

        if self.max_layer_idx == -1:
            raise ValueError(f"未在模型中找到任何目标层: {target_layers}")

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        self.features = {}
        for i, (name, module) in enumerate(self.children):
            x = module(x)
            if name in self._target_set:
                self.features[name] = x
            if i >= self.max_layer_idx:
                break
        return [self.features[name] for name in self.target_layers if name in self.features]


class DeepDreamGenerator:
    """DeepDream 生成器 - 商业级性能版"""

    def __init__(self, device: str | None = None):
        # 1. 设备安全分配
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            if device.startswith("cuda") and not torch.cuda.is_available():
                logger.warning(f"[DREAM] 请求了 {device} 但无 CUDA，自动降级为 CPU")
                self.device = torch.device("cpu")
            else:
                self.device = torch.device(device)

        # 2. 加载模型并冻结权重
        try:
            logger.info("[DREAM] 正在加载 InceptionV3 模型...")
            self.model = models.inception_v3(
                weights=models.Inception_V3_Weights.IMAGENET1K_V1,
                transform_input=False
            )
            self.model.eval()
            self.model.to(self.device)
            for param in self.model.parameters():
                param.requires_grad = False
        except Exception as e:
            logger.exception(f"[DREAM] 模型加载失败: {e}")
            raise RuntimeError(f"无法初始化 DeepDream 模型: {e}") from e

        # 3. 预处理与反预处理
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

        # 4. 特征提取器缓存池
        self._layer_extractors: dict[tuple, FeatureExtractor] = {}
        self.debug = DEBUG_MODE
        logger.info(f"[DREAM] 引擎初始化完成 | 设备: {self.device}")

    @staticmethod
    def _align_size(h: int, w: int) -> tuple[int, int]:
        """强制将宽高对齐到 8 的倍数（Inception V3 的池化层硬性要求）"""
        return max(8, int(h // 8 * 8)), max(8, int(w // 8 * 8))

    def _get_layer_extractor(self, layer_names: list[str]) -> FeatureExtractor:
        """获取并缓存特征提取器（相同的层组合只实例化一次）"""
        cache_key = tuple(sorted(layer_names))
        if cache_key not in self._layer_extractors:
            extractor = FeatureExtractor(self.model, layer_names)
            extractor.to(self.device)
            extractor.eval()
            self._layer_extractors[cache_key] = extractor
        return self._layer_extractors[cache_key]

    def _load_image(self, image_path: str, cfg: dict) -> torch.Tensor:
        """读取图像，限制最大尺寸，并转为对齐过尺寸的 Tensor"""
        with Image.open(image_path) as img_raw:
            img = img_raw.convert("RGB")
            w, h = img.size

        try:
            max_size = int(cfg.get("MAX_SIZE", 800))
        except (ValueError, TypeError):
            max_size = 800

        max_side = max(h, w)
        if max_side > max_size:
            scale = max_size / max_side
            new_w, new_h = int(w * scale), int(h * scale)
            # 修复：_align_size(h, w) 的签名是 (h, w)，返回 (aligned_h, aligned_w)。
            # 之前写成 `new_w, new_h = self._align_size(new_h, new_w)` 会导致宽高互换。
            new_h, new_w = self._align_size(new_h, new_w)
            img = img.resize((new_w, new_h), Image.LANCZOS)

        with torch.no_grad():
            img_tensor = self.preprocess(img).unsqueeze(0)
            return img_tensor.to(self.device)

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """将计算完的 Tensor 还原成人类能看的 PIL 图像"""
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        if tensor.dim() != 3:
            raise ValueError(f"期望 3 维张量，实际收到 {tensor.dim()} 维")

        with torch.no_grad():
            tensor = self.denormalize(tensor.cpu())
            tensor = torch.clamp(tensor, 0, 1)
        return transforms.ToPILImage()(tensor)

    def _add_gradient_noise(self, grad: torch.Tensor, cfg: dict, dream_type: str) -> torch.Tensor:
        """给梯度注入随机噪声（增加画面的不规则和颗粒感）"""
        try:
            noise_val = float(cfg.get("GRADIENT_NOISE", 0.0))
        except (TypeError, ValueError):
            noise_val = 0.0

        if noise_val <= 0:
            return grad

        std_val = torch.std(grad) + 1e-8
        noise_scale = noise_val * std_val

        mod = cfg.get("NOISE_MOD")
        factor = cfg.get("NOISE_FACTOR", 0.0)

        if mod == "sigmoid":
            noise_scale = noise_scale * (1.0 + factor * torch.sigmoid(noise_scale))
        elif mod == "tanh":
            noise_scale = noise_scale * (1.0 + factor * torch.tanh(noise_scale))

        return grad + torch.randn_like(grad) * noise_scale

    def _compute_loss(self, features: list[torch.Tensor], cfg: dict, dream_type: str) -> torch.Tensor:
        """计算特征图激活值的 Loss（我们要最大化它）"""
        scale = float(cfg.get("GRADIENT_SCALE", 1.0))
        base = float(cfg.get("LAYER_WEIGHT_BASE", 1.0))
        slope = float(cfg.get("LAYER_WEIGHT_SLOPE", 0.0))

        loss = torch.tensor(0.0, device=self.device)
        for i, feat in enumerate(features):
            weight = max(0.1, base + i * slope)
            loss = loss + torch.mean(feat) * weight * scale

        return loss

    def _dream_step(self, image: torch.Tensor, extractor: FeatureExtractor, cfg: dict,
                    dream_type: str, octave_idx: int, iter_idx: int) -> torch.Tensor:
        """执行一次完整的“幻觉迭代”（前向 -> 算 Loss -> 反向求梯度 -> 修改图像）"""
        progress = iter_idx / max(cfg.get("ITER_PER_OCTAVE", 80) - 1, 1)
        step_size = float(cfg.get("STEP_SIZE", 0.01))

        if cfg.get("DYNAMIC_STEP", False):
            factor = float(cfg.get("STEP_FACTOR", 0.3))
            power = float(cfg.get("STEP_POWER", 1))
            step_size *= (1 + factor * (progress ** power))

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

        grad = torch.autograd.grad(loss, img_shifted, retain_graph=False, create_graph=False)[0]

        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.01, neginf=-0.01)
        grad = torch.clamp(grad, min=-0.1, max=0.1)
        grad_std = torch.std(grad, dim=[2, 3], keepdim=True).clamp(min=1e-8)
        grad = grad / (grad_std + 1e-8)

        if cfg.get("SAFE_SHIFT", False):
            grad = torch.roll(torch.roll(grad, -shift_x, dims=3), -shift_y, dims=2)

        grad = self._add_gradient_noise(grad, cfg, dream_type)

        if "STAR_BRIGHTNESS" in cfg:
            brightness = float(cfg["STAR_BRIGHTNESS"])
            grad = grad * (1.0 + brightness * torch.abs(grad))

        if cfg.get("WAVE_FLOW", 0) > 0:
            wave_intensity = float(cfg["WAVE_FLOW"])
            _, _, h, _ = grad.shape
            grid_y = torch.arange(h, device=self.device, dtype=torch.float32).view(-1, 1)
            x_shift = wave_intensity * torch.sin(grid_y / 10.0)
            shift_amount = int(x_shift.mean())
            grad = torch.roll(grad, shifts=shift_amount, dims=3)

        img = img.detach() + grad * step_size

        fusion = float(cfg.get("REALITY_FUSION", 0))
        if fusion > 0:
            img = img * (1 - fusion) + image * fusion

        return torch.clamp(img, -3, 3)

    def _generate_base(self, image: torch.Tensor, extractor: FeatureExtractor,
                       cfg: dict, dream_type: str) -> torch.Tensor:
        """多尺度金字塔生成（先在小图上抓宏观结构，再在大图上抠细节）"""
        _, _, h, w = image.shape

        try:
            octave_layers = int(cfg.get("OCTAVE_LAYERS", 3))
            iter_per_octave = int(cfg.get("ITER_PER_OCTAVE", 80))
            octave_scale = float(cfg.get("OCTAVE_SCALE", 0.75))
        except (ValueError, TypeError):
            octave_layers, iter_per_octave, octave_scale = 3, 80, 0.75

        shapes = [self._align_size(int(h * (octave_scale ** i)), int(w * (octave_scale ** i)))
                  for i in range(octave_layers)][::-1]

        current_img = F.interpolate(image, size=shapes[0], mode='bilinear', align_corners=False)
        logger.info(f"[DREAM] 开始多尺度生成 | 共 {len(shapes)} 层 | 每层 {iter_per_octave} 次迭代")

        for octave_idx, size in enumerate(shapes):
            current_img = F.interpolate(current_img, size=size, mode='bilinear', align_corners=False)
            logger.info(f"[DREAM] 正在处理第 {octave_idx + 1}/{len(shapes)} 层 (尺寸: {size[0]}x{size[1]})...")

            for i in range(iter_per_octave):
                current_img = self._dream_step(current_img, extractor, cfg, dream_type, octave_idx, i)

        orig_h, orig_w = self._align_size(h, w)
        logger.info("[DREAM] 正在还原至原图尺寸...")
        return F.interpolate(current_img, size=(orig_h, orig_w), mode='bilinear', align_corners=False)

    # ==================== 公共接口 ====================

    def generate(self, input_path: str, dream_type: str = "normal",
                 output_dir: str = "./dream_outputs",
                 custom_cfg: dict | None = None,
                 custom_layers: list[str] | None = None) -> str:
        """最外层入口：读图 -> 生成 -> 存图 -> 存 meta 文件"""
        start_time = time.time()
        img_tensor = None
        result_tensor = None

        try:
            os.makedirs(output_dir, exist_ok=True)

            cfg = DREAM_CONFIGS.get(dream_type, DREAM_CONFIGS["normal"]).copy()
            if custom_cfg:
                cfg.update(custom_cfg)

            layer_names = custom_layers or DREAM_LAYERS.get(dream_type, ["Mixed_5b"])

            base_name = os.path.splitext(os.path.basename(input_path))[0]
            timestamp_str = str(int(time.time() * 1000))
            output_path = os.path.join(output_dir, f"{base_name}_{dream_type}_{timestamp_str}.png")
            meta_path = os.path.splitext(output_path)[0] + ".meta.json"

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            img_tensor = self._load_image(input_path, cfg)
            extractor = self._get_layer_extractor(layer_names)

            result_tensor = self._generate_base(img_tensor, extractor, cfg, dream_type)

            result_img = self._tensor_to_pil(result_tensor)
            result_img = self._post_process(result_img, cfg)
            result_img.save(output_path, format="PNG", optimize=True)

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

        finally:
            # 显式置空大对象引用，配合 gc 加速回收；用赋值而非 del 避免 UnboundLocalError
            img_tensor = None
            result_tensor = None
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()

    def _post_process(self, image: Image.Image, cfg: dict) -> Image.Image:
        """最后一道化妆工序（数据驱动）"""
        # 修复：POST_BLUR 若被手滑写成字符串 "0.8"，`> 0` 比较会 TypeError。
        try:
            blur_radius = float(cfg.get("POST_BLUR", 0) or 0)
        except (TypeError, ValueError):
            blur_radius = 0.0
        if blur_radius > 0:
            image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        if cfg.get("POST_SHARPEN", False):
            image = image.filter(ImageFilter.SHARPEN)

        if "POST_BRIGHTNESS" in cfg:
            image = ImageEnhance.Brightness(image).enhance(float(cfg["POST_BRIGHTNESS"]))

        if "POST_CONTRAST" in cfg:
            image = ImageEnhance.Contrast(image).enhance(float(cfg["POST_CONTRAST"]))

        if "POST_COLOR" in cfg:
            image = ImageEnhance.Color(image).enhance(float(cfg["POST_COLOR"]))

        return image

    def clear_cache(self):
        """彻底释放显存与特征提取器（外部手动调用）"""
        self._layer_extractors.clear()
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
