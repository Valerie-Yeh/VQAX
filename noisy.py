import json
import os
import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--nle_data", type=str,
                    help="nle_data/{VQA-X, aokvqa}/{vqaX, aokvqa}_{test, val}.json")
parser.add_argument("--img_dir", type=str,
                    help="val2014 or val2017")
parser.add_argument("--std", type=str,
                    help="decide what Gaussian std should be added")
args = parser.parse_args()

data = json.load(open(args.nle_data, 'r'))
data_k = list(data.keys())
img_list = []
for k in data_k:
    img_list.append(data[k]['image_name'])

# Custom transform to add Gaussian noise
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
img_transform = transforms.Compose([transforms.ToTensor(),
                                    AddGaussianNoise(0, float(args.std)),
                                    transforms.ToPILImage()])


if not os.path.isdir(f'images/noise{args.std}_{args.img_dir}'):
    os.makedirs(f'images/noise{args.std}_{args.img_dir}')
for img_file in tqdm(img_list):
    img = Image.open(f'images/{args.img_dir}/' + img_file).convert("RGB")
    img_augmented = img_transform(img)
    img_augmented.save(f'images/noise{args.std}_{args.img_dir}/' + img_file)