import torch
from torch.utils.data import Dataset, DataLoader
import os
from glob import glob
from tifffile import imread
import torchvision
from torchvision import transforms
from PIL import Image
from collections import Counter
import warnings
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torchvision import models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import csv
from torchvision.datasets import ImageFolder
import random
import torch.backends.cudnn as cudnn

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)  # for multi-GPU setups

# Disable CuDNN heuristics
cudnn.benchmark = False
cudnn.benchmark = False

# weights initialization
def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def cal_scores(targets, predictions, check=False):
  if check:
    true_labels = [int(t.item()) for t in targets]  # Extract integer values
    predicted_labels = [int(p.item()) for p in predictions]

  scores = []

  overall_accuracy = round(accuracy_score(true_labels, predicted_labels), 4)* 100

  # Calculate overall precision, recall, and F1-score (weighted average)
  overall_precision = round(precision_score(true_labels, predicted_labels, average='weighted'), 4)* 100
  overall_recall = round(recall_score(true_labels, predicted_labels, average='weighted'), 4)* 100
  overall_f1 = round(f1_score(true_labels, predicted_labels, average='weighted'), 4)* 100
  scores.extend([overall_accuracy, overall_precision, overall_recall, overall_f1])

  # Return dictionary containing all metrics
  return {
      'class_metrics': "class_metrics",
      'overall_accuracy': overall_accuracy,
      'overall_precision': overall_precision,
      'overall_recall': overall_recall,
      'overall_f1': overall_f1
  }, scores


# Evaluation function 
def evaluate_model(model, data_loader, lf):
    model.eval()
    model.to(device)
    saving_string = ""
    correct = 0
    total = 0
    predictions = []
    targets = []
    val_loss = 0.0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            # print(data.shape, target.shape)
            output = model(data)
            loss = lf(output, target)
            val_loss += loss
            _, predicted = torch.max(output.data, 1)
            predictions.extend(predicted)
            targets.extend(target)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    avg_val_loss = val_loss / len(data_loader)
    saving_string += f"Accuracy: {accuracy:.2f}% \n"
    print(f"Accuracy: {accuracy:.2f}%")
    print()
    dicrt, scores = cal_scores(predictions=predictions, targets=targets, check=True)
    print(dicrt)
    return saving_string, scores, avg_val_loss

class VGGModel_p(nn.Module):
  def __init__(self, pretrained=True):
    super(VGGModel_p, self).__init__()
    self.features = models.vgg16(pretrained=pretrained).features  # Use VGG16 features
    
    for param in self.features.parameters():
      param.requires_grad = False  # Freeze pre-trained layers
    
    self.avgpool = nn.AdaptiveAvgPool2d((7, 7))  # Global Average Pooling
    self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 3)  # 3 output classes
        )

  def forward(self, x):
    x = self.features(x)
    # print(x.shape)
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    # print(x.shape)
    x = self.classifier(x)
    return x
  
class VGGModel_ms(nn.Module):
  def __init__(self, pretrained=True):
    super(VGGModel_ms, self).__init__()
    vgg = models.vgg16(pretrained=pretrained)
    self.features = vgg.features  # Use VGG16 features
    
    for param in self.features.parameters():
      param.requires_grad = False  # Freeze pre-trained layers
    
    # Extract features from different layers of VGG
    self.features1 = nn.Sequential(*list(self.features.children())[:17])
    self.features2 = nn.Sequential(*list(self.features.children())[17:24])
    self.features3 = nn.Sequential(*list(self.features.children())[24:])

    # Upsampling layers for feature maps from lower layers
    self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    self.upsample3 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

    # Convolutional layers to process upsampled feature maps
    self.conv1 = nn.Conv2d(512, 256, kernel_size=1)
    self.conv2 = nn.Conv2d(512, 256, kernel_size=1)
    
    self.avgpool = nn.AdaptiveAvgPool2d((7, 7))  # Global Average Pooling
    self.classifier = nn.Sequential(
            nn.Linear(147 * 16 * 16, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 3)  # 3 output classes
        )

  def forward(self, x):
    # print(x.shape)
    # Extract features at different scales
    x1 = self.features1(x)
    # print(x1.shape)
    x2 = self.features2(x1)
    # print(x2.shape)
    x3 = self.features3(x2)

    # Upsample feature maps from lower layers
    x2_up = self.upsample2(x2)
    x3_up = self.upsample3(x3)

    # Process upsampled feature maps
    x2_up = self.conv1(x2_up)
    x3_up = self.conv2(x3_up)

    # print(x2_up.shape)
    # print(x3_up.shape)
    # print(x1.shape)

    # Concatenate feature maps from different scales
    fused_features = torch.cat((x1, x2_up, x3_up), dim=1)

    # x = self.features(x)
    # print(x.shape)
    x = self.avgpool(fused_features)
    x = torch.flatten(x, 1)
    # print(x.shape)
    x = self.classifier(x)
    return x
  

# base_line_sim = "3class/non_aug/splited_a"
# bicubic = "high_data/BICUBIC/all_sar_splited"
# bicubic = "high_data/BICUBIC/all_sar_splited"
# bilinear = "high_data/BILINEAR/all_sar_splited"
# lanczos = "high_data/LANCZOS/all_sar_splited"
# nearest = "high_data/NEAREST/all_sar_splited"
# Deep learning
edsr = "high_data/EDSR/all_sar_splited"
rcan = "high_data/RCAN/all_sar_splited"
carn = "high_data/CARN/all_sar_splited"

# baseline_vv = "3class/non_aug/vv_splited_b"
# baseline_vh = "3class/non_aug/vh_splited_b"

# Define data transformations
transform_train = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# simp_sar_train = ImageFolder(root= base_line_sim+"/train", transform=transform_train)
# # simp_sar_aug_train = ImageFolder(root=base_line_aug+ "/train", transform=transform_train)
# bicubic_train = ImageFolder(root=bicubic+ "/train", transform=transform_train)
# bilinear_train = ImageFolder(root=bilinear+ "/train", transform=transform_train)
# lanczos_train = ImageFolder(root=lanczos+ "/train", transform=transform_train)
# nearest_train = ImageFolder(root=nearest+ "/train", transform=transform_train)

edsr_train = ImageFolder(root=edsr+ "/train", transform=transform_train)
rcan_train = ImageFolder(root=rcan+ "/train", transform=transform_train)
carn_train = ImageFolder(root=carn+ "/train", transform=transform_train)


# simp_sar_val = ImageFolder(root=base_line_sim+ "/val", transform=transform_train)

# bicubic_val = ImageFolder(root=bicubic+ "/val", transform=transform_train)
# bilinear_val = ImageFolder(root=bilinear+ "/val", transform=transform_train)
# lanczos_val = ImageFolder(root=lanczos+ "/val", transform=transform_train)
# nearest_val = ImageFolder(root=nearest+ "/val", transform=transform_train)

edsr_val = ImageFolder(root=edsr+ "/val", transform=transform_train)
rcan_val = ImageFolder(root=rcan+ "/val", transform=transform_train)
carn_val = ImageFolder(root=carn+ "/val", transform=transform_train)


# simp_sar_test = ImageFolder(root=base_line_sim+ "/test", transform=transform_train)
# bicubic_test = ImageFolder(root=bicubic+ "/test", transform=transform_train)
# bilinear_test = ImageFolder(root=bilinear+ "/test", transform=transform_train)
# lanczos_test = ImageFolder(root=lanczos+ "/test", transform=transform_train)
# nearest_test = ImageFolder(root=nearest+ "/test", transform=transform_train)


edsr_test = ImageFolder(root=edsr+ "/test", transform=transform_train)
rcan_test = ImageFolder(root=rcan+ "/test", transform=transform_train)
carn_test = ImageFolder(root=carn+ "/test", transform=transform_train)

# simp_data_loader_train = DataLoader(simp_sar_train, batch_size=32, shuffle=False,
#                         num_workers=2,  # Experiment with different values as recommended above
#                         # pin_memory=False, # if torch.cuda.is_available() else False,
#                         persistent_workers=True)
# simp_data_loader_val = DataLoader(simp_sar_val, batch_size=32, shuffle=False,
#                         num_workers=2,  # Experiment with different values as recommended above
#                         # pin_memory=False, # if torch.cuda.is_available() else False,
#                         persistent_workers=True)
# simp_data_loader_test = DataLoader(simp_sar_test, batch_size=32, shuffle=False,
#                         num_workers=2,  # Experiment with different values as recommended above
#                         # pin_memory=False, # if torch.cuda.is_available() else False,
#                         persistent_workers=True)

# bicubic_train_loader = DataLoader(bicubic_train, batch_size=32, shuffle=False,
#                         num_workers=2,  # Experiment with different values as recommended above
#                         # pin_memory=False, # if torch.cuda.is_available() else False,
#                         persistent_workers=True)
# bicubic_loader_val = DataLoader(bicubic_val, batch_size=32, shuffle=False,
#                         num_workers=2,  # Experiment with different values as recommended above
#                         # pin_memory=False, # if torch.cuda.is_available() else False,
#                         persistent_workers=True)
# bicubic_loader_test = DataLoader(bicubic_test, batch_size=32, shuffle=False,
#                         num_workers=2,  # Experiment with different values as recommended above
#                         # pin_memory=False, # if torch.cuda.is_available() else False,
#                         persistent_workers=True)

# bilinear_train_loader = DataLoader(bilinear_train, batch_size=32, shuffle=False,
#                         num_workers=2,  # Experiment with different values as recommended above
#                         # pin_memory=False, # if torch.cuda.is_available() else False,
#                         persistent_workers=True)
# bilinear_loader_val = DataLoader(bilinear_val, batch_size=32, shuffle=False,
#                         num_workers=2,  # Experiment with different values as recommended above
#                         # pin_memory=False, # if torch.cuda.is_available() else False,
#                         persistent_workers=True)
# bilinear_loader_test = DataLoader(bilinear_test, batch_size=32, shuffle=False,
#                         num_workers=2,  # Experiment with different values as recommended above
#                         # pin_memory=False, # if torch.cuda.is_available() else False,
#                         persistent_workers=True)

# lanczos_train_loader = DataLoader(lanczos_train, batch_size=32, shuffle=False,
#                         num_workers=2,  # Experiment with different values as recommended above
#                         # pin_memory=False, # if torch.cuda.is_available() else False,
#                         persistent_workers=True)
# lanczos_loader_val = DataLoader(lanczos_val, batch_size=32, shuffle=False,
#                         num_workers=2,  # Experiment with different values as recommended above
#                         # pin_memory=False, # if torch.cuda.is_available() else False,
#                         persistent_workers=True)
# lanczos_loader_test = DataLoader(lanczos_test, batch_size=32, shuffle=False,
#                         num_workers=2,  # Experiment with different values as recommended above
#                         # pin_memory=False, # if torch.cuda.is_available() else False,
#                         persistent_workers=True)

# nearest_train_loader = DataLoader(nearest_train, batch_size=32, shuffle=False,
#                         num_workers=2,  # Experiment with different values as recommended above
#                         # pin_memory=False, # if torch.cuda.is_available() else False,
#                         persistent_workers=True)
# nearest_loader_val = DataLoader(nearest_val, batch_size=32, shuffle=False,
#                         num_workers=2,  # Experiment with different values as recommended above
#                         # pin_memory=False, # if torch.cuda.is_available() else False,
#                         persistent_workers=True)
# nearest_loader_test = DataLoader(nearest_test, batch_size=32, shuffle=False,
#                         num_workers=2,  # Experiment with different values as recommended above
#                         # pin_memory=False, # if torch.cuda.is_available() else False,
#                         persistent_workers=True)


# DL loaders
edsr_train_loader = DataLoader(edsr_train, batch_size=32, shuffle=False,
                        num_workers=2,  # Experiment with different values as recommended above
                        # pin_memory=False, # if torch.cuda.is_available() else False,
                        persistent_workers=True)
edsr_loader_val = DataLoader(edsr_val, batch_size=32, shuffle=False,
                        num_workers=2,  # Experiment with different values as recommended above
                        # pin_memory=False, # if torch.cuda.is_available() else False,
                        persistent_workers=True)
edsr_loader_test = DataLoader(edsr_test, batch_size=32, shuffle=False,
                        num_workers=2,  # Experiment with different values as recommended above
                        # pin_memory=False, # if torch.cuda.is_available() else False,
                        persistent_workers=True)

rcan_train_loader = DataLoader(rcan_train, batch_size=32, shuffle=False,
                        num_workers=2,  # Experiment with different values as recommended above
                        # pin_memory=False, # if torch.cuda.is_available() else False,
                        persistent_workers=True)
rcan_loader_val = DataLoader(rcan_val, batch_size=32, shuffle=False,
                        num_workers=2,  # Experiment with different values as recommended above
                        # pin_memory=False, # if torch.cuda.is_available() else False,
                        persistent_workers=True)
rcan_loader_test = DataLoader(rcan_test, batch_size=32, shuffle=False,
                        num_workers=2,  # Experiment with different values as recommended above
                        # pin_memory=False, # if torch.cuda.is_available() else False,
                        persistent_workers=True)

carn_train_loader = DataLoader(carn_train, batch_size=32, shuffle=False,
                        num_workers=2,  # Experiment with different values as recommended above
                        # pin_memory=False, # if torch.cuda.is_available() else False,
                        persistent_workers=True)
carn_loader_val = DataLoader(carn_val, batch_size=32, shuffle=False,
                        num_workers=2,  # Experiment with different values as recommended above
                        # pin_memory=False, # if torch.cuda.is_available() else False,
                        persistent_workers=True)
carn_loader_test = DataLoader(carn_test, batch_size=32, shuffle=False,
                        num_workers=2,  # Experiment with different values as recommended above
                        # pin_memory=False, # if torch.cuda.is_available() else False,
                        persistent_workers=True)



l_rate = 0.0001
epochs = 30
lf = nn.CrossEntropyLoss()

# datasets = {"non_aug": [[simp_data_loader_train, simp_data_loader_val],simp_data_loader_test],
#             "OpenSAR_aug": [[aug_data_loader_train, aug_data_loader_val], aug_data_loader_test]}
# datasets = {"OpenSAR_aug": [[aug_data_loader_train, aug_data_loader_val], aug_data_loader_test]}
# datasets = {"OpenSAR_all": [[simp_data_loader_train, simp_data_loader_val], simp_data_loader_test],
#             "bicubic": [[bicubic_train_loader, bicubic_loader_val], bicubic_loader_test],
#             "bilinear": [[bilinear_train_loader, bilinear_loader_val], bilinear_loader_test],
#             "lanczos": [[lanczos_train_loader, lanczos_loader_val], lanczos_loader_test],
#             "nearest": [[nearest_train_loader, nearest_loader_val], nearest_loader_test]}

datasets = {"edsr": [[edsr_train_loader, edsr_loader_val], edsr_loader_test],
            "rcan": [[rcan_train_loader, rcan_loader_val], rcan_loader_test],
            "carn": [[carn_train_loader, carn_loader_val], carn_loader_test]}

models = {
          "VGG_pretrained": VGGModel_p(),
          "VGG_multiScale": VGGModel_ms()
          }


results = ""
csv_res = []
csv1 = []

for model_name, n_model in models.items():
   results += "Training on " + model_name + "\n"
   for dataset_name, dataset_loader in datasets.items():
        print("Training on ", dataset_name)
        results += "Training on " + dataset_name + "\n"
        # mix_path = "mix_5"
        train_loader_m = dataset_loader[0][0]
        val_loader_m = dataset_loader[0][1]
        test_loader_m = dataset_loader[1]
        n_model.apply(initialize_weights)

        # n_model = VGGModel_ms()
        optimizer = optim.Adam(n_model.parameters(), lr=l_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        # train_model(n_model, train_loader_m, val_loader_m, lf, optimizer, epochs, 5)

        n_model.to(device)
        best_val_loss = float('inf')
        epochs_no_improve = 0

        # Training loop
        for epoch in range(epochs):
            n_model.train()
            print(epoch)
            running_loss = 0.0
            for batch_idx, data in enumerate(train_loader_m):
                data, target = data[0].to(device), data[1].to(device)
                # data, target = data[0], data[1]

                optimizer.zero_grad()

                output = n_model(data)

                loss = lf(output, target)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                if batch_idx % 15 == 0:
                    results+=f'Epoch {epoch + 1}/{epochs}, Batch {batch_idx}/{len(train_loader_m)}, Loss: {loss.item()}\n'
                    # print(f'Epoch {epoch + 1}/{epochs}, Batch {batch_idx}/{len(train_loader_m)}, Loss: {loss.item()}')
            avg_train_loss = running_loss / len(train_loader_m)
            str_results, csv_scores, avg_val_loss = evaluate_model(n_model, val_loader_m, lf)
            results += "Validation \n"
            results += str_results
            scheduler.step(avg_val_loss)
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                torch.save(n_model.state_dict(), 'best_model.pth')  # Save the best model
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= 20:
                    print('Early stopping!')
                    break
            

        print("Evaluating: ", dataset_name)
        results += f"Testing on {dataset_name} \n"
        n_model.load_state_dict(torch.load('best_model.pth'))
        str_results, csv_scores, _ = evaluate_model(n_model, test_loader_m, lf)
        results += str_results
        csv_res.append(csv_scores)
        csv1.append(csv_scores)

# Open the file in write mode ("w") and write the string to it
with open("sr_results_dl.txt", "w") as f:
  f.write(results)

fields = ["Accuracy", "Precision", "Recall", "F1"]


with open('sr_scores_dl.csv', 'w') as f:

    # using csv.writer method from CSV package
    write = csv.writer(f)

    write.writerow(fields)
    write.writerows(csv1)


# import os
# import shutil
# import random

# def split_images_by_class(data_dir, split_dir, train_size=0.8, val_size=0.1):
#   """
#   Splits images by class into train, validation, and test sets.

#   Args:
#     data_dir: Path to the directory containing the image data (with class subfolders).
#     split_dir: Path to the directory where the split data will be created.
#     train_size: Proportion of data for training (0-1).
#     val_size: Proportion of data for validation (0-1).
#   """

#   # Create the split directory if it doesn't exist
#   if not os.path.exists(split_dir):
#     os.makedirs(split_dir)

#   # Get a list of class names
#   class_names = os.listdir(data_dir)
  
#   train_dir = os.path.join(split_dir, 'train')
#   val_dir = os.path.join(split_dir, 'val')
#   test_dir = os.path.join(split_dir, 'test')

#   for class_name in class_names:
#     class_dir = os.path.join(data_dir, class_name)
#     # split_class_dir = os.path.join(split_dir, class_name)
#     s_train_dir = os.path.join(train_dir, class_name)
#     s_val_dir = os.path.join(val_dir, class_name)
#     s_test_dir = os.path.join(test_dir, class_name)

#     # Create class-specific split directories
#     os.makedirs(train_dir, exist_ok=True)
#     os.makedirs(val_dir, exist_ok=True)
#     os.makedirs(test_dir, exist_ok=True)

#     os.makedirs(s_train_dir, exist_ok=True)
#     os.makedirs(s_val_dir, exist_ok=True)
#     os.makedirs(s_test_dir, exist_ok=True)

#     # Get a list of image files in the class directory
#     image_files = os.listdir(class_dir)

#     # Shuffle the image files randomly
#     random.shuffle(image_files)

#     # Calculate sizes for train, validation, and test sets
#     num_images = len(image_files)
#     train_num = int(train_size * num_images)
#     val_num = int(val_size * num_images)

#     print(class_name, train_num, val_num)

#     # train_num = train_size
#     # val_num = val_size

#     # Split the image files into train, validation, and test sets
#     train_files = image_files[:train_num]
#     val_files = image_files[train_num:train_num+val_num]
#     test_files = image_files[train_num+val_num:]

#     # Copy images to respective directories
#     for file in train_files:
#       src_file = os.path.join(class_dir, file)
#       dst_file = os.path.join(s_train_dir, file)
#       shutil.copy(src_file, dst_file)

#     for file in val_files:
#       src_file = os.path.join(class_dir, file)
#       dst_file = os.path.join(s_val_dir, file)
#       shutil.copy(src_file, dst_file)

#     for file in test_files:
#       src_file = os.path.join(class_dir, file)
#       dst_file = os.path.join(s_test_dir, file)
#       shutil.copy(src_file, dst_file)

# for idx, method in enumerate(["EDSR", "RCAN", "CARN"]):
#   data_dir = 'high_data/' + method + "/" + "all_sar"
#   split_dir = 'high_data/' + method + "/" + "all_sar_splited"

#   split_images_by_class(data_dir, split_dir)

#   data_dir = 'high_data/' + method + "/" + "pols/vv"
#   split_dir = 'high_data/' + method + "/" + "pols/vv_splited"

#   split_images_by_class(data_dir, split_dir)

#   data_dir = 'high_data/' + method + "/" + "pols/vh"
#   split_dir = 'high_data/' + method + "/" + "pols/vh_splited"

#   split_images_by_class(data_dir, split_dir)