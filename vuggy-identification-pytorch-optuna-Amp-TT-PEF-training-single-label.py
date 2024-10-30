# %% [markdown]
# # Deep-Learning-Based Vuggy Facies Identification from Borehole Images
# Source: https://www.researchgate.net/publication/344755725_Deep-Learning-Based_Vuggy_Facies_Identification_from_Borehole_Images  
# 
# O artigo comenta uma separação do conjunto de treinamento em 80% pro treino, 10% pra validação e 10% pro teste. 
# 
# ## Ideias:
#  - Testar buscar uma das classes bem, talvez juntar duas classes em uma para ver o que acontece... coisas assim!
#  - Será que rola usar um modelo auto-supervisionado nesse caso e explorar as possibilidades?

# %%
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import re
import os
import random
# from functools import partial

import torch
import torchvision as tv
import torchmetrics as tm

from torch import nn
from torch.nn import functional as F
# from torchvision import datasets, models, transforms
import torch.optim as optim
from torch.utils.data import Dataset

from skimage import io
from PIL import Image
# from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
# from sklearn import metrics
from tqdm import tqdm
from time import perf_counter, sleep

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import optuna

# from torchmetrics import ConfusionMatrix
from sklearn.metrics import confusion_matrix
# import seaborn as sn

import argparse

display = print # migueeeee!!

# %%
# for corte in [10]:
#     print(f'\n\n# Corte {corte}')
#     pattern = r"(\d-LL-\d{2,3})-(2-cortada)-RJS-(\d{1,4}).png"
#     basepath = f'./dataset/7-LL-69/Static/PT2/Corte {corte}'
#     files = next(os.walk(basepath))[2]
#     files = [f for f in files if f[-3:] == 'png']
#     files.sort()

#     for file in files:
#         well, part, index = re.match(pattern, file).groups()
#         # print(well, part, index)
#         print(' - ', file, '->', f'{well}-RJS-{part}-{index}.png')
#         # raise Exception()
#         os.rename(f'{basepath}/{file}', f'{basepath}/{well}-RJS-{part}-{index}.png')

# %%
# Change names for an entire folder at once
# trocar os nomes dos arquivos para o mesmo em todas as pastas, tanto amplitude quanto TT!
# o nome será 7-LL-69-RJS-1-010
# pattern = r"(.+-RJS)-(\d{1,2})-.+-(\d{1,3})"

# for i in tqdm(range(1, 11)):
#     path = f'./Segmentadas/TT/PT{i}'
#     files = next(os.walk(path))[2]
#     # print(files)
#     for file in files:
#         re_match = re.match(pattern=pattern, string=file).groups()
#         new_file = f'{re_match[0]}-{re_match[1]:0>2}-TT-{re_match[2]:0>3}.png'
#         # print(file, new_file)
#         # os.rename(path+'/'+file, path+'/'+new_file)
    

# %%
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

# Set seeds for every lib
set_seed(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device', device)

# %%
# Open the dataset
# Sizes with formatting as x,y
image_sizes = {
    10:  (1152,  118), 
    # 20:  (1152,  236), 
    40:  (1152,  472), 
    # 60:  (1152,  708), 
    70: (1152, 826),
    # 80:  (1152,  944), 
    100: (1152, 1180) 
}

input_infos = ['AW', 'TT']

wells_basepaths = {
    '3-BRSA-1201': './dataset/3-BRSA-1201',
    '7-LL-11':  './dataset/7-LL-11',
    '7-LL-69':  './dataset/7-LL-69',
    '8-LL-112': './dataset/8-LL-112',
    '8-LL-92':  './dataset/8-LL-92',
    '9-BRSA-1254':  './dataset/9-BRSA-1254',
}

CLASS_NAMES = [
        'Non-Vug',                          # 0
        'Incipient Matrix Dissolution',     # 1
        'Incipient Fracture Dissolution',   # 2
        'Intense Matrix Dissolution',       # 3
        'Intense Fracture Dissolution',     # 4
        'Well Developed Karst'              # 5
    ]
TARGET = 'User Class'
TABULAR_COLUMNS = []

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--cut_height', type=int, default=10)
parser.add_argument('--reduction_factor', type=int, default=3)
parser.add_argument('--epochs', type=int, default=800)
parser.add_argument('--early_stop', type=int, default=100)
parser.add_argument('--n_studies', type=int, default=300)
args = parser.parse_args()

batch_size = args.batch_size
selected_cut_height = args.cut_height
reduction_factor = args.reduction_factor

# %%
image_size = (int(image_sizes[selected_cut_height][1]/reduction_factor), int(image_sizes[selected_cut_height][0]/reduction_factor))
# image_size = (224, 224)
print(f'{image_size=}')

# %% [markdown]
# ## Load the dataset
# In the case where each label table is inside the path, use the following cell.

# %%
columns = ['Filename-Static', 'Filename-Travel-Time', TARGET]
dataset = []

# Each dataset cell below could be used depending on the chosen file structure
# %%
# for well_name, well_path in wells_basepaths.items():
#     # check if there are two parts
#     if 'PT1' in next(os.walk(f'{well_path}/Static'))[1]:
#         df_list = []
#         for pt in next(os.walk(f'{well_path}/Static'))[1]:
#             df = pd.read_excel(f'{well_path}/Static/{pt}/Corte {selected_cut_height}/{well_name}-{pt}-{selected_cut_height}.xlsx', header=1, names=columns)
#             df['Filename'] = df['Filename'].apply(lambda x: f'{well_path}/Static/{pt}/Corte {selected_cut_height}/{x}.png')
#             df_list.append(df)
#         df = pd.concat(df_list, axis='rows')
#     else:
#         df = pd.read_excel(f'{well_path}/Static/Corte {selected_cut_height}/{well_name}-{selected_cut_height}.xlsx', header=1, names=columns)
#         df['Filename'] = df['Filename'].apply(lambda x: f'{well_path}/Static/Corte {selected_cut_height}/{x}.png')

#     dataset = pd.concat((dataset, df), axis='rows')
#     dataset.reset_index(inplace=True, drop=True)

# %%
for well_name, well_path in wells_basepaths.items():
    #df = pd.read_excel(f'{well_path}/{well_name}-CNN_CLASS-{selected_cut_height}cm.xlsx', header=1, names=columns)
    df = pd.read_excel(f'{well_path}/{well_name}-CNN_CLASS-{selected_cut_height}cm(FINAL).xlsx')
    df = df[columns]

    df['Filename-Static'] = df['Filename-Static'].apply(
        lambda x: f'{well_path}/Static/Corte {selected_cut_height}/{x}'
    )

    df['Filename-Travel-Time'] = df['Filename-Travel-Time'].apply(
        lambda x: f'{well_path}/Travel Time/Corte {selected_cut_height}/{x}'
    )

    dataset.append(df)
dataset = pd.concat(dataset, axis=0)
dataset.reset_index(inplace=True, drop=True)

# %% [markdown]
# ## Other way to load the dataset
# Use this if you have one table with all labels, with each well in a separate sheet with the well name.

# %%

print(dataset)
# Add new patches to the dataset, only if 100
if selected_cut_height == 100:
    new_100 = pd.read_excel('./dataset/Classes para treino 100cm/new-patches-ch-100.xlsx')
    new_100 = new_100[['Filename-Static', 'Filename-Travel-Time', 'User Class']]
    # dataset = pd.concat((dataset, new_100), axis=0)
# %%
print(dataset)

# %%
dataset.dropna(inplace=True)
dataset.reset_index(inplace=True, drop=True)
# dataset['Class'] = dataset['Class'].astype(int)
dataset = dataset[dataset['User Class'] < 6]
print(dataset)

dataset.to_excel(f'./dataset/dataset-all-ch-{selected_cut_height}cm(FINAL).xlsx')
train_df, test_df = train_test_split(dataset, test_size=.20, random_state=42, stratify=dataset['User Class'])

if selected_cut_height == 100:
    train_df = pd.concat((train_df, new_100), axis=0)

# %%
# train_unique = np.unique(train_df['Class'], return_counts = True)
# test_unique = np.unique(test_df['Class'], return_counts = True)

# fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8,3))

# axs[0].set_title('Training set')
# bars = axs[0].bar(train_unique[0], height = train_unique[1])
# axs[0].bar_label(bars)
# axs[0].set_xticks([i for i in range(6)])
# axs[0].set_ylim(0, 1.1*np.max(train_unique[1]))

# axs[1].set_title('Validation set')
# bars = axs[1].bar(test_unique[0], height = test_unique[1])
# axs[1].bar_label(bars)
# axs[1].set_xticks([i for i in range(6)])
# axs[1].set_ylim(0, 1.1*np.max(test_unique[1]))

# plt.tight_layout()
# plt.show()

# %%
pattern = r"(.+-RJS)-(\d{1,2})-.+-(\d{1,3})"

def find_TT_filename(fname):
   fname_parts = fname.split('/')
   
   fn = fname_parts[-1]
   fpath = '/'.join(fname_parts[:-1])

   try:
      files = next(os.walk(fpath))[2]
   except StopIteration:
      print('Deu erro na pasta:', fpath, fname)

   try:
      f = [f'{fpath}/{f}' for f in files if fn.split('-')[-1] == f.split('-')[-1]][0]
   except IndexError:
      print(fpath, fn)
      raise Exception()
   return f

class ImagelogDataset(Dataset):
   def __init__(
               self, 
               df, 
               target, 
               class_names, 
               tabular_columns,
               phase,
               inputs=['AW', 'TT', 'Tabular'], 
               dtype_min_max=(0, 255), 
               min_max = (-1, 1), 
               target_size = (108, 352), 
               train_augmentation=False):
               
      self.df = df
      self.class_names = class_names
      self.phase = phase
      self.min_max = min_max
      self.dtype_min_max = dtype_min_max
      self.train_augmentation = train_augmentation
      self.tabular_columns = tabular_columns
      self.target = target
      self.inputs = inputs

      if 'AW' in inputs:
         self.images_amp = [np.array(Image.open(fn).resize(size=target_size[::-1], resample=Image.Resampling.BICUBIC)) for fn in self.df['Filename-Static']] #  .transpose((2,0,1))
         for img_idx in range(len(self.images_amp)):
            if self.images_amp[img_idx].shape[-1] == 4:
               self.images_amp[img_idx] = self.images_amp[img_idx][:, :, :3]

      if 'TT' in inputs:
         self.images_tt  = [np.array(Image.open(fn).resize(size=target_size[::-1], resample=Image.Resampling.BICUBIC)) for fn in self.df['Filename-Travel-Time']] #  .transpose((2,0,1))
         for img_idx in range(len(self.images_tt)):
            if self.images_tt[img_idx].shape[-1] == 4:
               self.images_tt[img_idx] = self.images_tt[img_idx][:, :, :3]

      if 'Tabular' in inputs:
         self.tabular = self.df[tabular_columns]#.to_list()

      self.classes =  self.df[target]#.to_list()

      self.data_len = len(self.classes)
        
   def __len__(self):
      return self.data_len

   def transform(self, image, p=.5, rect_sizes_min_max=[(.05, .05), (.40, .40)]):
      image = Image.fromarray(image)

      # ColorJitter
      # if random.random() < p:
      #    brightness  = random.choice(np.linspace(0.8, 1.2, 5))
      #    contrast    = random.choice(np.linspace(0.8, 1.2, 5))
      #    saturation  = random.choice(np.linspace(0.8, 1.2, 5))
      #    sharpness   = random.choice(np.linspace(0.8, 1.2, 5))

      #    image = tv.transforms.functional.adjust_brightness(image, brightness)
      #    image = tv.transforms.functional.adjust_contrast(image, contrast)
      #    image = tv.transforms.functional.adjust_saturation(image, saturation)
      #    image = tv.transforms.functional.adjust_sharpness(image, sharpness)

      # RandomHorizontalFlip
      if random.random() < p:
         image = tv.transforms.functional.hflip(image)

      # # RandomVerticalFlip
      if random.random() < p:
         image = tv.transforms.functional.vflip(image)

      image = np.array(image)
      if random.random() < p:
         top    = np.random.randint(0, int((1-rect_sizes_min_max[1][0])*image.shape[0]))
         left   = np.random.randint(0, int((1-rect_sizes_min_max[1][1])*image.shape[1]))
         height = np.random.randint(int(rect_sizes_min_max[0][0]*image.shape[0]), int(rect_sizes_min_max[1][0]*image.shape[0]))
         width  = np.random.randint(int(rect_sizes_min_max[0][1]*image.shape[1]), int(rect_sizes_min_max[1][1]*image.shape[1]))
         image[top:top+height, left:left+width] = 0

      return image

   def normalize(self, img):
      # 0 to 1
      image = (img - self.dtype_min_max[0])/(self.dtype_min_max[1]-self.dtype_min_max[0])
      # -1 to 1
      image = image * (self.min_max[1] - self.min_max[0]) + self.min_max[0]

      return image

   def __getitem__(self, index):
      item = {'index': index, 'class': np.array(self.classes.iloc[index])}

      if 'AW' in self.inputs:
         image_amp = self.images_amp[index]
         if self.train_augmentation:
            image_amp = self.transform(image_amp)
         image_amp = self.normalize(image_amp)
         item['AW'] = image_amp.transpose((2,0,1))

      if 'TT' in self.inputs:
         image_tt = self.images_tt[index]
         if self.train_augmentation:
            image_tt  = self.transform(image_tt)
         image_tt  = self.normalize(image_tt)
         item['TT'] = image_tt.transpose((2,0,1))

      if 'Tabular' in self.inputs:
         item['Tabular'] = self.tabular.iloc[index]

      return item

def get_vision_model(
               model=tv.models.mobilenet_v3_small,
               weights='DEFAULT',
               last_layer=3,
               n_filters=32,
               n_layers=1,
            ):

    # create model
    model = model(weights=weights)

    if 'AW' in input_infos and 'TT' in input_infos:
        in_channels = 6
    else:
        in_channels = 3

    # freeze the layers
    if weights:
        for param in model.parameters():
            param.requires_grad = False

    # Modify the first layer to accept 3 or 6 channels
    if hasattr(model, 'features'):
        first_layer = model.features[0][0]
        model.features[0][0] = nn.Conv2d(in_channels, first_layer.out_channels,  kernel_size=first_layer.kernel_size, stride=first_layer.stride, padding=first_layer.padding, bias=False)
    if hasattr(model, 'stem'):
        first_layer = model.stem[0]
        model.stem[0] = nn.Conv2d(in_channels, first_layer.out_channels,  kernel_size=first_layer.kernel_size, stride=first_layer.stride, padding=first_layer.padding, bias=False)
    elif hasattr(model, 'conv1'):
        if type(model.conv1) == nn.modules.container.Sequential:
            first_layer = model.conv1[0]
            model.conv1[0] = nn.Conv2d(in_channels, first_layer.out_channels,  kernel_size=first_layer.kernel_size, stride=first_layer.stride, padding=first_layer.padding, bias=False)
        elif type(model.conv1) == type(tv.models.googlenet().conv1):
            first_layer = model.conv1.conv
            model.conv1.conv = nn.Conv2d(in_channels, first_layer.out_channels,  kernel_size=first_layer.kernel_size, stride=first_layer.stride, padding=first_layer.padding, bias=False)
        else:
            first_layer = model.conv1
            model.conv1 = nn.Conv2d(in_channels, first_layer.out_channels,  kernel_size=first_layer.kernel_size, stride=first_layer.stride, padding=first_layer.padding, bias=False)
    elif hasattr(model, 'layers'):
        if type(model.layers) == nn.modules.container.Sequential:
            first_layer = model.layers[0]
            model.layers[0] = nn.Conv2d(in_channels, first_layer.out_channels,  kernel_size=first_layer.kernel_size, stride=first_layer.stride, padding=first_layer.padding, bias=False)
        else:
            first_layer = model.layers
            model.layers = nn.Conv2d(in_channels, first_layer.out_channels,  kernel_size=first_layer.kernel_size, stride=first_layer.stride, padding=first_layer.padding, bias=False)
    elif hasattr(model, 'heads'):
        number_features = model.heads.head.out_features
        features = list(model.heads.children())
        model.conv_proj = nn.Conv2d(in_channels, 768, kernel_size=(16, 16), stride=(16, 16))
        # model.conv_proj.in_channels = in_channels

    # Modify the last layer
    if hasattr(model, 'classifier'):
        number_features = model.classifier[last_layer].in_features
        features = list(model.classifier.children())[:-1] # Remove last layer
    elif hasattr(model, 'fc'):
        number_features = model.fc.in_features
        features = []

    if n_layers == 0:
        # features.append(torch.nn.Linear(number_features, len(CLASS_NAMES)))
        features.append(torch.nn.Linear(number_features, n_filters))
    else:
        features.extend([
                torch.nn.Linear(number_features, n_filters),
                torch.nn.ReLU(),
            ])

        for _ in range(n_layers-1):
            features.extend([
                torch.nn.Linear(n_filters, n_filters),
                torch.nn.ReLU(),
            ])

        features.append(torch.nn.Linear(n_filters, n_filters))

    if hasattr(model, 'classifier'):
        model.classifier = torch.nn.Sequential(*features)
    elif hasattr(model, 'fc'):
        model.fc = torch.nn.Sequential(*features)
    elif hasattr(model, 'heads'):
        model.heads = torch.nn.Sequential(*features)

    # model = model.to(device)

    return model

class VuggyClassifier(nn.Module):
   def __init__(self, 
            inputs,
            model=tv.models.mobilenet_v3_small,
            weights=tv.models.MobileNet_V3_Small_Weights.DEFAULT,
            last_layer=3,
            n_filters=32,
            n_layers=1,
            tabular_filters=8,
            dropout_ratio=.3):

      super(VuggyClassifier, self).__init__()

      self.inputs = inputs
      self.image = get_vision_model(model=model, weights=weights, last_layer=last_layer, n_filters=n_filters, n_layers=n_layers)
      
      self.tabular = nn.Sequential(
            nn.Linear(len(TABULAR_COLUMNS), tabular_filters),
            nn.ReLU(),
      )

      input_filters = 0
      if 'AW' in self.inputs or 'TT' in self.inputs:
         input_filters += n_filters
      if 'Tabular' in self.inputs:
         input_filters += tabular_filters
      
      layers = [nn.Linear(input_filters, n_filters), nn.LeakyReLU(), nn.Dropout(p=dropout_ratio), nn.BatchNorm1d(num_features=n_filters)]
      for _ in range(n_layers):
         layers.extend([
               nn.Linear(n_filters, n_filters),
               nn.LeakyReLU(),
         ])
      layers.append(nn.Linear(n_filters, len(CLASS_NAMES)))

      self.classifier = nn.Sequential(
         *layers
         # nn.Softmax() # the CrossEntropyLoss already has the softmax function built int ;D
      )

   def forward(self, x): # AW, TT, Tabular
      inputs = []

      if 'AW' in self.inputs and 'TT' in self.inputs:
         inputs.append(self.image(torch.cat((x['AW'], x['TT']), 1)))
      elif 'AW' in self.inputs:
         inputs.append(self.image(x['AW']))
      elif 'TT' in self.inputs:
         inputs.append(self.image(x['TT']))
      
      if 'Tabular' in self.inputs:
         inputs.append(self.tabular(x['Tabular']))

      if type(inputs[0]) == tv.models.GoogLeNetOutputs:
          inputs[0] = inputs[0].logits
      
      x = torch.cat(inputs, dim=1)

      x = self.classifier(x)
      # x = F.softmax(x, dim=0) # The CrossEntropyLoss has already a softmax layer
      # x = F.sigmoid(x) # only with BCELoss to have the probabilities. The BCEWithLogitsLoss already has a sigmoid layer.

      return x

# %%
dataset_train = ImagelogDataset(
    train_df, 
    target=TARGET, 
    class_names=CLASS_NAMES, 
    tabular_columns=TABULAR_COLUMNS, 
    phase='train', 
    inputs = input_infos,
    target_size=image_size, 
    train_augmentation=False
)
dataset_test  = ImagelogDataset(
    test_df,  
    target=TARGET, 
    class_names=CLASS_NAMES, 
    tabular_columns=TABULAR_COLUMNS, 
    phase='test',
    inputs = input_infos,
    target_size=image_size, 
    train_augmentation=False
)
dataset_all   = ImagelogDataset(
    dataset,  
    target=TARGET, 
    class_names=CLASS_NAMES, 
    tabular_columns=TABULAR_COLUMNS, 
    phase='test', 
    inputs = input_infos,
    target_size=image_size
)

dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=False, num_workers=4)
dataloader_test  = torch.utils.data.DataLoader(dataset_test,  batch_size=batch_size, shuffle=False, num_workers=4)
dataloader_all   = torch.utils.data.DataLoader(dataset_all,   batch_size=batch_size, shuffle=False, num_workers=4)

# %%
def get_checkpoint_prefix(input_infos):
    prefixes = []
    if 'AW' in input_infos:
        prefixes.append('AW')
    if 'TT' in input_infos:
        prefixes.append('TT')
    if 'Tabular' in input_infos:
        prefixes.append('-'.join(TABULAR_COLUMNS))
    
    return '-'.join(prefixes)
get_checkpoint_prefix(input_infos)

# %%
num_epochs = args.epochs
early_stopping_epochs = args.early_stop
class_prob_threshold = .05

threshold_dict = {
    10: {
        'accuracy': 0.5,
        'f1': 0.5,
        'auroc': 0.85
    },
    40: {
        'accuracy': 0.5,
        'f1': 0.5,
        'auroc': 0.85
    },
    70: {
        'accuracy': 0.5,
        'f1': 0.5,
        'auroc': 0.8
    },
    100: {
        'accuracy': 0.4,
        'f1': 0.4,
        'auroc': 0.8
    },
}

do_augmentation = True
rect_sizes_min_max=[(.05, .05), (.40, .40)]
augmentation_prob = .5

checkpoint_path = f'./checkpoints-{selected_cut_height}-single-label-final'
if not os.path.exists(checkpoint_path):
   os.mkdir(checkpoint_path)

def image_augmentation(image_aw, image_tt, p=.5, color_min=.9, color_max=1.1):
    # image -> tensor

    # ColorJitter
    if random.random() < p:
        brightness  = random.choice(np.linspace(color_min, color_max, 5))
        contrast    = random.choice(np.linspace(color_min, color_max, 5))
        saturation  = random.choice(np.linspace(color_min, color_max, 5))
        sharpness   = random.choice(np.linspace(color_min, color_max, 5))

        image_aw = tv.transforms.functional.adjust_brightness(image_aw, brightness)
        image_aw = tv.transforms.functional.adjust_contrast(image_aw, contrast)
        image_aw = tv.transforms.functional.adjust_saturation(image_aw, saturation)
        image_aw = tv.transforms.functional.adjust_sharpness(image_aw, sharpness)

        image_tt = tv.transforms.functional.adjust_brightness(image_tt, brightness)
        image_tt = tv.transforms.functional.adjust_contrast(image_tt, contrast)
        image_tt = tv.transforms.functional.adjust_saturation(image_tt, saturation)
        image_tt = tv.transforms.functional.adjust_sharpness(image_tt, sharpness)

    # RandomHorizontalFlip
    if random.random() < p:
        image_aw = tv.transforms.functional.hflip(image_aw)
        image_tt = tv.transforms.functional.hflip(image_tt)

    # # RandomVerticalFlip
    if random.random() < p:
        image_aw = tv.transforms.functional.vflip(image_aw)
        image_tt = tv.transforms.functional.vflip(image_tt)

    # random region removal
    # image = np.array(image)
    # if random.random() < p:
    #     top    = np.random.randint(0, int((1-rect_sizes_min_max[1][0])*image_aw.shape[0]))
    #     left   = np.random.randint(0, int((1-rect_sizes_min_max[1][1])*image_aw.shape[1]))
    #     height = np.random.randint(int(rect_sizes_min_max[0][0]*image_aw.shape[0]), int(rect_sizes_min_max[1][0]*image_aw.shape[0]))
    #     width  = np.random.randint(int(rect_sizes_min_max[0][1]*image_aw.shape[1]), int(rect_sizes_min_max[1][1]*image_aw.shape[1]))

    #     image_aw[top:top+height, left:left+width] = image_aw.mean()
    #     image_tt[top:top+height, left:left+width] = image_tt.mean()

    # random azimuth rotation - interesting to see if works!
    if random.random() < p:
        shift_step = 10
        shift = random.choice(np.arange(shift_step, image_aw.shape[-1], shift_step))
        
        image_aw = torch.roll(image_aw, shift, dims=-1)
        image_tt = torch.roll(image_tt, shift, dims=-1)
    
    return image_aw, image_tt

model_dict = {
    'efficientnet_B0':      {'model': tv.models.efficientnet_b0,    'last_layer': 1, 'weights': {None:None, 'DEFAULT': tv.models.EfficientNet_B0_Weights.DEFAULT}},
    # 'googlenet':            {'model': tv.models.googlenet,          'last_layer': 0, 'weights': {None:None, 'DEFAULT': tv.models.GoogLeNet_Weights.DEFAULT}},
    # 'mnasnet_0_5':          {'model': tv.models.mnasnet0_5,         'last_layer': 1, 'weights': {None:None, 'DEFAULT': tv.models.MNASNet0_5_Weights.DEFAULT}},
    # 'mnasnet_0_75':         {'model': tv.models.mnasnet0_75,        'last_layer': 1, 'weights': {None:None, 'DEFAULT': tv.models.MNASNet0_75_Weights.DEFAULT}},
    # 'mnasnet_1_0':          {'model': tv.models.mnasnet1_0,         'last_layer': 1, 'weights': {None:None, 'DEFAULT': tv.models.MNASNet1_0_Weights.DEFAULT}},
    # 'mobilenet_v2':         {'model': tv.models.mobilenet_v2,       'last_layer': 1, 'weights': {None:None, 'DEFAULT': tv.models.MobileNet_V2_Weights.DEFAULT}},
    'mobilenet_v3_small':   {'model': tv.models.mobilenet_v3_small, 'last_layer': 3, 'weights': {None:None, 'DEFAULT': tv.models.MobileNet_V3_Small_Weights.DEFAULT}},
    # 'mobilenet_v3_large':   {'model': tv.models.mobilenet_v3_large, 'last_layer': 3, 'weights': {None:None, 'DEFAULT': tv.models.MobileNet_V3_Large_Weights.DEFAULT}},
    # # 'regnet_y_400mf':       {'model': tv.models.regnet_y_400mf,     'last_layer': 0, 'weights': {None:None, 'DEFAULT': tv.models.RegNet_Y_400MF_Weights.DEFAULT}},
    # 'regnet_y_800mf':       {'model': tv.models.regnet_y_800mf,     'last_layer': 0, 'weights': {None:None, 'DEFAULT': tv.models.RegNet_Y_800MF_Weights.DEFAULT}},
    # 'regnet_x_400mf':       {'model': tv.models.regnet_x_400mf,     'last_layer': 0, 'weights': {None:None, 'DEFAULT': tv.models.RegNet_X_400MF_Weights.DEFAULT}},
    # 'regnet_x_800mf':       {'model': tv.models.regnet_x_800mf,     'last_layer': 0, 'weights': {None:None, 'DEFAULT': tv.models.RegNet_X_800MF_Weights.DEFAULT}},
    'resnet18':             {'model': tv.models.resnet18,           'last_layer': 0, 'weights': {None:None, 'DEFAULT': tv.models.ResNet18_Weights.DEFAULT}},
    # # 'shufflenet_v2_x0_5':   {'model': tv.models.shufflenet_v2_x0_5, 'last_layer': 0, 'weights': {None:None, 'DEFAULT': tv.models.ShuffleNet_V2_X0_5_Weights.DEFAULT}},
    # 'shufflenet_v2_x1_0':   {'model': tv.models.shufflenet_v2_x1_0, 'last_layer': 0, 'weights': {None:None, 'DEFAULT': tv.models.ShuffleNet_V2_X1_0_Weights.DEFAULT}},
    # 'shufflenet_v2_x1_5':   {'model': tv.models.shufflenet_v2_x1_5, 'last_layer': 0, 'weights': {None:None, 'DEFAULT': tv.models.ShuffleNet_V2_X1_5_Weights.DEFAULT}},
    # 'shufflenet_v2_x2_0':   {'model': tv.models.shufflenet_v2_x2_0, 'last_layer': 0, 'weights': {None:None, 'DEFAULT': tv.models.ShuffleNet_V2_X2_0_Weights.DEFAULT}},
    
    # 'squeezenet_1_0':       {'model': tv.models.squeezenet1_0,      'last_layer': 1, 'weights': {None:None, 'DEFAULT': tv.models.SqueezeNet1_0_Weights.DEFAULT}}, # Esse modelo e o de baixo possuem uma Conv2d como classifier... vale a pena ajustar?
    # 'squeezenet_1_1':       {'model': tv.models.squeezenet1_1,      'last_layer': 1, 'weights': {None:None, 'DEFAULT': tv.models.SqueezeNet1_1_Weights.DEFAULT}},
}

def objective_fn(trial):
    # weights_list = [None, 'DEFAULT']

    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=False, num_workers=4)
    dataloader_test  = torch.utils.data.DataLoader(dataset_test,  batch_size=batch_size, shuffle=False, num_workers=4)
    # dataloader_all   = torch.utils.data.DataLoader(dataset_all,   batch_size=batch_size, shuffle=False, num_workers=4)

    weights_label = trial.suggest_categorical('weights', [None]) # , 'DEFAULT', 'IMAGENET'
    n_filters=2**trial.suggest_int('n_filters', 1,8) #1, 10)
    n_layers=trial.suggest_int('n_layers', 1, 5)
    # tabular_filters = trial.suggest_int('tabular_filters', 1, 1)

    model_name = trial.suggest_categorical('model', list(model_dict.keys()))
    model = VuggyClassifier(
        inputs=input_infos,
        model=model_dict[model_name]['model'], 
        weights=model_dict[model_name]['weights'][weights_label], 
        # last_group=model_dict[model_name]['last_group'], 
        last_layer=model_dict[model_name]['last_layer'], 
        n_filters=n_filters, 
        n_layers=n_layers, 
        # freeze=freeze
    )

    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)

    optimizer_name = trial.suggest_categorical('optimizer', ['SGD', 'Adam'])
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)

    pbar = tqdm(range(num_epochs), desc='Training model')

    best = {
        'epoch': -1, 'accuracy': 0, 'f1': 0, 'f1-lbl':0, 'ev': 0, 'auroc': -1, 'loss':1000, 'mse':1e5
    }

    history = {
        'epoch': [], 'train-loss':[], 'validation-loss': [], 'accuracy-macro': [], 'f1-macro': [],
        'accuracy-weighted': [], 'f1-weighted': [], 'auroc-macro': [], 'auroc-weighted': []
    }
    history = history | {f'f1-class-{c}': [] for c in range(6)}
    history = history | {f'accuracy-class-{c}': [] for c in range(6)}
    history = history | {f'auroc-class-{c}': [] for c in range(6)}

    for epoch in pbar:
        # Iterate over data.
        train_loss = 0
        model.train()
        for data in dataloader_train:
            inputs_dict = {}
            if 'AW' in data.keys():
                inputs_dict['AW'] = data['AW'].float().to(device)
            if 'TT' in data.keys():
                inputs_dict['TT'] = data['TT'].float().to(device)
            if 'Tabular' in data.keys():
                inputs_dict['Tabular'] = data['Tabular'].float().to(device)

            labels = data['class']
            labels = labels.to(device) #.type(torch.long)

            if do_augmentation:
                augmentation_prob = trial.suggest_float('augmentation_p', .1, .9, step=.1)
                inputs_dict['AW'], inputs_dict['TT'] = image_augmentation(inputs_dict['AW'], inputs_dict['TT'], p=augmentation_prob)

            # print(labels.dtype, labels.shape)
            # raise Exception()

            optimizer.zero_grad()
            
            # with torch.set_grad_enabled(True):
            outputs  = model(inputs_dict)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # validate model
        val_loss = 0
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for data in dataloader_test:
                inputs_dict = {}
                if 'AW' in data.keys():
                    inputs_dict['AW'] = data['AW'].float().to(device)
                if 'TT' in data.keys():
                    inputs_dict['TT'] = data['TT'].float().to(device)
                if 'Tabular' in data.keys():
                    inputs_dict['Tabular'] = data['Tabular'].float().to(device)

                labels = data['class']
                labels = labels.type(torch.long).to(device)
                
                # with torch.set_grad_enabled(True):
                outputs  = model(inputs_dict)
                loss = criterion(outputs, labels)

                # _, preds = torch.max(outputs, 1)
                # preds = preds.cpu().numpy()
                preds = outputs.cpu().numpy()
                labels = labels.cpu().numpy()

                # print(f'{preds.shape=}, {labels.shape=}, {y_pred=}, {y_true=}')
                y_pred.extend(preds)
                y_true.extend(labels)
                # print(f'{y_pred=}, {y_true=}')
                # raise Exception()

                val_loss += loss.item()
    
        # y_true, y_pred = np.array(y_true), np.array(y_pred)
        y_true_t, y_pred_t = torch.Tensor(y_true).long(), torch.Tensor(y_pred)
        _, y_pred_best = torch.max(y_pred_t, 1)
        

        params = {
            'task': 'multiclass',
            'num_classes': 6,
        }
        targets = {

             'preds': y_pred_t,
            'target': y_true_t,
        }

        all_accuracy = tm.Accuracy(**params, average=None)(**targets).cpu().numpy()
        all_f1 = tm.F1Score(**params, average=None)(**targets).cpu().numpy()
        all_auroc = tm.AUROC(**params, average=None)(target=y_true_t, preds=y_pred_t).cpu().numpy()

        val_accuracy = all_accuracy.mean()
        val_f1 = all_f1.mean()
        val_f1_lbl = all_f1[[2,4,5]].mean()
        val_auroc = all_auroc.mean()

        history['epoch'].append(epoch)
        history['train-loss'].append(train_loss)
        history['validation-loss'].append(val_loss)
        history['accuracy-macro'].append(val_accuracy)
        history['f1-macro'].append(val_f1)
        history['auroc-macro'].append(val_auroc)
        history['accuracy-weighted'].append(tm.Accuracy(**params, average='weighted')(**targets).cpu().numpy())
        history['f1-weighted'].append(tm.F1Score(**params, average='weighted')(**targets).cpu().numpy())
        history['auroc-weighted'].append(tm.AUROC(**params, average='weighted')(target=y_true_t, preds=y_pred_t).cpu().numpy())

        for c in range(params['num_classes']):
            history[f'accuracy-class-{c}'].append(all_accuracy[c])
            history[f'auroc-class-{c}'].append(all_auroc[c])
            history[f'f1-class-{c}'].append(all_f1[c])

        if val_f1 > best['f1'] and val_accuracy > best['accuracy']:
            best['epoch'] = epoch
            best['model_state_dict'] = model.state_dict()
            # best['report'] = metrics.classification_report(y_true_bin, y_pred_bin, zero_division=0)
            best['preds'] = y_pred
            best['true'] = y_true
            best['loss'] = val_loss

            best['accuracy'] = val_accuracy
            best['f1'] = val_f1
            best['f1-lbl'] = val_f1_lbl
            best['auroc'] = val_auroc

        # rudimentary early stopping
        if early_stopping_epochs > 0 and epoch - best['epoch'] > early_stopping_epochs:
            break

        desc  = f'Training model -> train_loss {train_loss:.4f} - '
        desc += f'val_loss {val_loss:.4f} - '
        desc += f"f1 {val_f1:.3f} -> "
        desc += f"Best so far: #{best['epoch']: >3} "
        desc += f"accuracy {best['accuracy']:.3f} "
        desc += f"f1 {best['f1']:.3f} "
        desc += f"f1-lbl {best['f1-lbl']:.3f} "
        desc += f"auroc {best['auroc']:.3f}"
        
        pbar.set_description(desc)

    if best['accuracy'] > threshold_dict[selected_cut_height]['accuracy'] and best['f1'] > threshold_dict[selected_cut_height]['f1'] and best['auroc'] > threshold_dict[selected_cut_height]['auroc']:
        file_path = f'{checkpoint_path}/model-{get_checkpoint_prefix(input_infos)}-{model_name}-{weights_label}-{optimizer_name}-lr-{lr:.6f}-n_filters-{n_filters}-n_layers-{n_layers}-bs-{batch_size}-rf-{reduction_factor}-ch-{selected_cut_height}-aug_p-{augmentation_prob:.2f}'
        file_name = f'-acc-{best["accuracy"]:.4f}-f1-{best["f1"]:.4f}-f1_lbl-{best["f1-lbl"]:.4f}-auroc-{best["auroc"]:.4f}-ep-{best["epoch"]}'
        print('Saving', file_path+file_name, end=' ...')

        history_df = pd.DataFrame.from_dict(data=history)
        history_df.to_excel(file_path+file_name+'.xlsx')
        torch.save(model, file_path+file_name+'.pth')
        print('Foi!')

    return best['accuracy'], best["f1"], best["auroc"] 

# %%
import warnings
warnings.filterwarnings("ignore") # Again, bad practice, but it helps to avoid infinite warning messages on the terminal

# %%
sampler = optuna.samplers.RandomSampler(seed=123)
pruner = optuna.pruners.MedianPruner()
study = optuna.create_study(directions=["maximize", "maximize", "maximize"], study_name="Vuggy Classifier", sampler=sampler, pruner=pruner) # "maximize", 
study.optimize(objective_fn, n_trials=args.n_studies)