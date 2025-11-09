# debug_backward.py
import torch
from torchvision import datasets, transforms
from liptrf.models.vit import ViT
import torch.nn as nn
import torch.optim as optim

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
])
ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)
data, target = next(iter(loader))

device = torch.device("cpu")
data, target = data.to(device), target.to(device)

model = ViT(image_size=32, patch_size=4, num_classes=10, channels=3,
            dim=192, depth=1, heads=3, mlp_ratio=4,
            attention_type="dp", dropout=0.1, lmbda=0.001, power_iter=10,
            device=device).to(device)
criterion = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=1e-4)

# one-step train with anomaly detection
torch.autograd.set_detect_anomaly(True)
out = model(data)
loss = criterion(out, target)
print("Loss finite:", torch.isfinite(loss).all().item(), "loss:", loss.item())
loss.backward()
# inspect grads
for name, p in model.named_parameters():
    if p.grad is None:
        continue
    if not torch.isfinite(p.grad).all():
        print("Non-finite grad:", name)
        print(" grad stats:", p.grad.min().item(), p.grad.max().item(), p.grad.mean().item(), torch.linalg.norm(p.grad).item())
        break
print("All gradients finite up to check")

