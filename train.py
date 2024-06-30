import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from cnnModel import CNNModel
from customDataset import CustomDataset
import time 

data_dir = 'data/images'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
noise_factor = 0.5
n_epochs = 50
batch_size = 16


transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = CustomDataset(data_dir, transform=transform)
dataset.setNoiseFactor(noise_factor)

data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print("Loaded Data")

model = CNNModel()
model = model.to(device).float()
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

def calculate_mse(outputs, targets):
    mse = torch.mean((outputs - targets) ** 2).item()
    return mse

start_time = time.time()

for epoch in range(n_epochs):
    for noisy_imgs, clean_imgs in data_loader:
        noisy_imgs, clean_imgs = noisy_imgs.to(device).float(), clean_imgs.to(device).float()
        
        # Forward pass
        outputs = model(noisy_imgs)
        loss = loss_fn(outputs, clean_imgs)

        mse = calculate_mse(outputs, clean_imgs)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}, MSE: {mse:.4f}")

print("Training complete!")

print(f"Time Taken at batch_size: {batch_size}: {time.time()-start_time}")


torch.save({
    'epoch': n_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'models/saved_model2.pt')