from torchvision import models
import torch

def main():
    # 1. 使用最新的 API 加载模型（避免警告）
    weights = models.Inception_V3_Weights.IMAGENET1K_V1
    model = models.inception_v3(weights=weights)
    model.eval()

    print("=== 顶层模块 (named_children) ===")
    # 这对应你代码里常用的 Mixed_5b, Conv2d_3b_1x1 等
    for name, module in model.named_children():
        print(f"- {name} (类型: {type(module).__name__})")

    print("\n=== 完整结构 (named_modules) 前 30 个 ===")
    # 如果你要查更深层的名字，看这里
    for i, (name, module) in enumerate(model.named_modules()):
        if i > 30:
            break
        print(f"- {name}")

if __name__ == "__main__":
    main()
