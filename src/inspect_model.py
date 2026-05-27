import torch
from depth_anything_3.api import DepthAnything3

model = DepthAnything3.from_pretrained("depth-anything/DA3MONO-LARGE")

print("=== named_children (top level) ===")
for name, _ in model.named_children():
    print(f"  {name}")

print("\n=== named_children (model.model) ===")
for name, _ in model.model.named_children():
    print(f"  {name}")

print("\n=== Linear layers (candidate LoRA targets) ===")
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        print(f"  {name}  [{module.in_features} -> {module.out_features}]")
