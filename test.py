import os
import torch
import cv2
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

model = torch.load('KidsegUnet_dice_new.pt')
model = model.cuda()
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return image

def save_mask(mask, save_path):
    cv2.imwrite(save_path, (mask * 255))

def predict_and_save(input_dir, output_dir):
    # Define acceptable image extensions
    valid_extensions = ('.jpg', '.jpeg', '.png')

    # Count the number of image files to set up the progress bar correctly
    total_files = sum([len([f for f in files if f.lower().endswith(valid_extensions)]) for r, d, files in os.walk(input_dir)])

    with tqdm(total=total_files, desc="Processing Images", unit="image") as pbar:
        for root, _, files in sorted(os.walk(input_dir)):

            for file in files:
                if file.lower().endswith(valid_extensions):
                    image_path = os.path.join(root, file)
                    save_path = os.path.join(output_dir, os.path.relpath(image_path, input_dir))
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                    # Load image
                    image = load_image(image_path)
                    image = t(image)
                    model.to(device)
                    image = image.to(device)
                    with torch.no_grad():
                        image = image.unsqueeze(0)
                        output = model(image)
                        masked = torch.argmax(output, dim=1)
                        masked = masked.cpu().squeeze(0).numpy()

                    # Save the mask
                    save_mask(masked, save_path)
                    
                    # Update the progress bar
                    pbar.update(1)


# import os
# import torch
# import cv2
# from torchvision import transforms
# from PIL import Image
# import numpy as np

# model = torch.load('KidsegUnet_dice_new.pt')
# model = model.cuda()
# model.eval()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# mean=[0.485, 0.456, 0.406]
# std=[0.229, 0.224, 0.225]

# t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

# def load_image(image_path):
#     image = Image.open(image_path).convert('RGB')
#     return image

# def save_mask(mask, save_path):
#     cv2.imwrite(save_path,(mask * 255))
#     # mask_image.save(save_path)

# def predict_and_save(input_dir, output_dir):
#     for root, _, files in sorted(os.walk(input_dir)):
#         for file in files:
#             if file.endswith('.jpg'):
#                 image_path = os.path.join(root, file)
#                 save_path = os.path.join(output_dir, os.path.relpath(image_path, input_dir))
#                 os.makedirs(os.path.dirname(save_path), exist_ok=True)

#                 # Load image
#                 image = load_image(image_path)

#                 image = t(image)
#                 model.to(device); image=image.to(device)
#                 with torch.no_grad():
        
#                     image = image.unsqueeze(0)
                    
#                     output = model(image)
#                     print(output.shape)

#                     masked = torch.argmax(output, dim=1)
#                     masked = masked.cpu().squeeze(0).numpy()

#                 # Save the mask
#                 save_mask(masked, save_path)