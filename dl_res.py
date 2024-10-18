import pickle
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor, ToPILImage
# import matplotlib.pyplot as plt
from PIL import Image
# import cv2
import os
import warnings
from collections import defaultdict
from torchsr.models import edsr, rcan, carn
from torch.utils.data import Dataset, DataLoader
import math


warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# List of models to iterate through
# models = {
#     'EDSR': [edsr(scale=2, pretrained=True), edsr(scale=4, pretrained=True)],
#     'RCAN': [rcan(scale=2, pretrained=True), rcan(scale=4, pretrained=True)],
#     'CARN': [carn(scale=2, pretrained=True), carn(scale=4, pretrained=True)]
# }

models = {
    'EDSR': [edsr(scale=2, pretrained=True)],
    'RCAN': [rcan(scale=2, pretrained=True)],
    'CARN': [carn(scale=2, pretrained=True)]
}

# Move models to the device once, and freeze them
for idx, (model_name, model) in enumerate(models.items()):
    model[0].to(device).eval()  # Move 2x model to device
    # model[1].to(device).eval()  # Move 4x model to device
    for param in model[0].parameters():
        param.requires_grad = False
    # for param in model[1].parameters():
    #     param.requires_grad = False

main_path = "3class/non_aug/"
storing_data = "high_data/"
polarizations = ["vv", "vh"]
classes = ["Cargo", "Fishing", "Tanker"]

scores = defaultdict(lambda: defaultdict(dict))

def resize_to_nearest_multiple_of_4(image):
    # Get original image size
    original_width, original_height = image.size
    
    # Calculate the nearest multiples of 4
    new_width = math.ceil(original_width / 4) * 4
    new_height = math.ceil(original_height / 4) * 4

    # Resize the image using the new dimensions
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return resized_image

# Optimized Transformations
to_tensor = ToTensor()
to_pil_image = ToPILImage()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for pol in polarizations:
    print(pol)
    for cl in classes:
        print(cl)
        p_path = os.path.join(main_path, pol, cl)
        images_list = os.listdir(p_path)
        t_images = len(images_list)

        # score_ssim_2x = [0, 0, 0]
        # score_psnr_2x = [0, 0, 0]
        # score_ssim_4x = [0, 0, 0]
        # score_psnr_4x = [0, 0, 0]
        for idx, (model_name, model) in enumerate(models.items()):
            model[0].to(device)  # Move models to device once, outside of image loop
            # model[1].to(device)
            model[0].eval()
            # model[1].eval()
            for i_image in images_list:
                image_path = os.path.join(p_path, i_image)

                try:
                    image_hr = Image.open(image_path).convert('RGB')
                    # image_hr = resize_to_nearest_multiple_of_4(image_hr)
                except Exception as e:
                    print(f"Error opening image {image_path}: {e}")
                    continue


                original_width, original_height = image_hr.size
                
                if original_width < 2 or original_height < 2:
                    print("Image too small to downscale:", image_path)
                    continue
                
                # image_lr_2 = image_hr.resize((original_width // 2, original_height // 2), Image.Resampling.LANCZOS)
                # lr_tensor_2 = ToTensor()(image_lr_2).unsqueeze(0)
                # image_lr_4 = image_hr.resize((original_width // 4, original_height // 4), Image.Resampling.LANCZOS)
                # lr_tensor_4 = ToTensor()(image_lr_4).unsqueeze(0)
                # image_hr_np = np.array(image_hr)

                hr_tensor = ToTensor()(image_hr).unsqueeze(0)
                
                # Inside the image processing loop
                with torch.no_grad():
                    sr_tensor2 = model[0](hr_tensor.to(device))
                    # sr_tensor4 = model[1](lr_tensor_4.to(device))

                sr_image2 = ToPILImage()(sr_tensor2.squeeze())
                # sr_image4 = ToPILImage()(sr_tensor4.squeeze())


                hr_path = os.path.join(storing_data, model_name, "all_sar", cl)
                
                os.makedirs(hr_path, exist_ok=True)
                sr_image2.save(os.path.join(hr_path, i_image))
                # vh_image_up_2.save(os.path.join(hr_path, i_image.replace(".tif", "_vh.tif")))

        #         sr_image2_np = np.array(sr_image2)
        #         sr_image4_np = np.array(sr_image4)

        #         # print(image_hr_np.shape, sr_image2_np.shape)

        #         psnr_value_2 = np.round(psnr(image_hr_np, sr_image2_np), 2)
        #         score_psnr_2x[idx] += psnr_value_2

        #         ssim_value_2 = np.round(ssim(image_hr_np, sr_image2_np, win_size=3, channel_axis=-1), 4)
        #         score_ssim_2x[idx] += ssim_value_2

        #         psnr_value_4 = np.round(psnr(image_hr_np, sr_image4_np), 2)
        #         score_psnr_4x[idx] += psnr_value_4

        #         ssim_value_4 = np.round(ssim(image_hr_np, sr_image4_np, win_size=3, channel_axis=-1), 4)
        #         score_ssim_4x[idx] += ssim_value_4
        
        # # Normalize the scores first to avoid redundant np.array() and division operations
        # ssim_2x_normalized = list(np.array(score_ssim_2x) / t_images)
        # psnr_2x_normalized = list(np.array(score_psnr_2x) / t_images)
        # ssim_4x_normalized = list(np.array(score_ssim_4x) / t_images)
        # psnr_4x_normalized = list(np.array(score_psnr_4x) / t_images)

        # # Assign the values to the respective dictionary keys
        # scores[pol][cl]['ssim_2x'] = ssim_2x_normalized
        # scores[pol][cl]['psnr_2x'] = psnr_2x_normalized
        # scores[pol][cl]['ssim_4x'] = ssim_4x_normalized
        # scores[pol][cl]['psnr_4x'] = psnr_4x_normalized

# # Convert the defaultdict to a regular dict
# scores_dict = {pol: {cl: dict(values) for cl, values in classes.items()} for pol, classes in scores.items()}

# # Save the scores dictionary to a file
# with open('super_resolution_scores_dict.pkl', 'wb') as file:
#     pickle.dump(scores_dict, file)
