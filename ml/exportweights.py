import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(28*28, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)


state_dict = torch.load("tinyml_weights.pth")


model.load_state_dict(state_dict, strict=False)
model.eval()

print("Layers in state_dict:")
for key in state_dict.keys():
    print(key)

layer_order = ["1.weight", "1.bias", "3.weight", "3.bias"]

with open("weights.mem", "w") as f:
    for name in layer_order:
        param = state_dict[name]
        for p in param.flatten():
            f.write(f"{p.item()}\n")

print("Weights exported to weights.mem successfully!")
