import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import cv2
import torch
import urllib.request

import matplotlib.pyplot as plt

filepath = r"./Data/Dataset1/"
filename = "image_10.jpeg"
output_name="depth_10.jpg"

model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform
img = cv2.imread(filepath+filename)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

input_batch = transform(img).to(device)

with torch.no_grad():
    prediction = midas(input_batch)

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

output = prediction.cpu().numpy()

# Display the depth map
plt.imshow(output, cmap='inferno')  # You can choose any colormap you prefer
plt.colorbar()  # Add a color bar to show the depth scale
plt.axis('off')  # Turn off axis
plt.title('Depth Map')
plt.show()

output = cv2.normalize(output, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
cv2.imwrite(filepath + output_name,output)