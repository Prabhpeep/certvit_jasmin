# debug_forward.py
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from liptrf.models.vit import ViT

def print_stats(name, t):
    t = t.detach().float()
    print(f"[{name}] finite={torch.isfinite(t).all().item()} min={t.min().item():.4e} max={t.max().item():.4e} mean={t.mean().item():.4e} std={t.std().item():.4e} norm={torch.linalg.norm(t).item():.4e}")

def main():
    device = torch.device("cpu")
    # tiny transform: same as your training but keep it simple
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
    ])
    ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)
    data, target = next(iter(loader))
    data, target = data.to(device), target.to(device)

    model = ViT(image_size=32, patch_size=4, num_classes=10, channels=3,
                dim=192, depth=1, heads=3, mlp_ratio=4,
                attention_type="dp", dropout=0.1, lmbda=0.001, power_iter=10,
                device=device).to(device)
    model.eval()

    # attach forward hooks that inspect outputs of each module but are quiet unless problem
    bad_found = {"flag": False}

    def mk_hook(name):
        def hook(m, inp, out):
            if bad_found["flag"]:
                return
            # convert out to tensor(s)
            if isinstance(out, (tuple, list)):
                tensors = out
            else:
                tensors = (out,)
            for i, t in enumerate(tensors):
                if not torch.is_tensor(t):
                    continue
                if not torch.isfinite(t).all():
                    print(f"[BAD] Module {name} produced non-finite tensor (index {i})")
                    print_stats(f"{name}.out[{i}]", t)
                    bad_found["flag"] = True
                    # after first bad tensor, dump module param stats
                    for pname, p in m.named_parameters():
                        print(f"    param: {pname}, finite={torch.isfinite(p).all().item()}, norm={torch.linalg.norm(p).item():.4e}")
                    raise RuntimeError("Non-finite output detected")
        return hook

    handles = []
    for name, module in model.named_modules():
        # skip trivial container modules
        if len(list(module.children())) == 0:
            handles.append(module.register_forward_hook(mk_hook(name)))

    # run forward with anomaly detection to catch backward NaNs if they happen
    torch.autograd.set_detect_anomaly(True)
    try:
        out = model(data)
        print("Model forward completed. final output stats:")
        print_stats("model_out", out)
    except Exception as e:
        print("Exception during forward:", repr(e))
    finally:
        for h in handles:
            h.remove()

if __name__ == "__main__":
    main()

