import torch
from torchvision import transforms
from cnnModel import CNNModel
import cv2
import numpy as np
from torchview import draw_graph

model_path = 'models/saved_model.pt'
input_image_path = 'data/images/17.png'
noise_factor = 0.2

model = CNNModel()
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model.to('cuda').float()

image = cv2.imread(input_image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = image/255.0

noisy_image = image + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=image.shape)

cv2.imshow('input', noisy_image)
cv2.waitKey(0)

image_tensor = transforms.ToTensor()(noisy_image).unsqueeze(0).cuda()
image_tensor = image_tensor.to('cuda').float()

with torch.no_grad():
    output = model(image_tensor)

output = output.squeeze().cpu().numpy()

cv2.imshow('result', output)
cv2.waitKey(0)

# model_graph = draw_graph(model, input_size=(1, 1, 28, 28), expand_nested=True)
# model_graph.visual_graph.render("attached", format="png")