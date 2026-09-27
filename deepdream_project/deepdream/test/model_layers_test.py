import torchvision.models as models

model = models.inception_v3(pretrained=True)
for name, _ in model.named_children():
    print(name)
