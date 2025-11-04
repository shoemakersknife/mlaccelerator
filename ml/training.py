import torch
from torch import nn
from torchvision import datasets, transforms

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

# Define simple network
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28, 16),
    nn.ReLU(),
    nn.Linear(16, 10),
    nn.LogSoftmax(dim=1)
)

# Training setup
loss_fn = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Train (very quick — just to get usable weights)
for epoch in range(3):
    for images, labels in train_loader:
        optimizer.zero_grad()
        output = model(images)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch} complete, loss={loss.item():.4f}")

# Save trained weights for hardware inference later
torch.save(model.state_dict(), "tinyml_weights.pth")
print("Model trained and weights saved!")
