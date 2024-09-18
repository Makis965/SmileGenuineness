import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pandas as pd
from torchvision.io import read_image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class MaskTransform:

    def __call__(self, img):

        for i in range(img.size()[0]):
            h, w = img.size()[2], img.size()[3]
            mask_size = int(0.35 * h)  

            x = np.random.randint(0, h - mask_size)
            y = np.random.randint(0, w - mask_size)
            
            img[i, :, x:x+mask_size, y:y+mask_size] = 0 
    
        return img

def to_float32(tensor):
    return tensor.float()/ 255

img_transforms = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.Lambda(to_float32),
])

def load_files_from_dir(root_dir, df: pd.DataFrame):
    '''Iterate over directories to get images' paths'''
    if "celeb" in root_dir:
        df_attr = pd.read_csv('list_attr_celeba.txt', sep='\s+', skiprows=1)
        df_attr['Sample'] = df_attr.index
        df_attr['Sample'] = root_dir + "/" + df_attr['Sample']
        df_attr = df_attr.reset_index()
        df_attr['Smiling'] = df_attr['Smiling'].replace(-1, 0)
        df_attr['Male'] = df_attr['Male'].replace(-1, 0)
        df_attr = df_attr[["Sample", "Male", "Smiling"]]
        
        return df_attr
    
def split_sets(dataset_path = "dataset.csv"):
    '''Args:
        dataset_path (String): path to dataset stored in .CSV file
       containing columns = ["Sample", "Male", "Smiling"]'''
       
    df = pd.read_csv(dataset_path)

    train_X, test_X, train_y, test_y = train_test_split(df["Sample"], df[["Male", "Smiling"]], 
                                test_size=0.01, random_state=42, shuffle=True, 
                                stratify=df[["Male", "Smiling"]])

    train = pd.concat([train_X, train_y], axis=1)
    test = pd.concat([test_X, test_y], axis=1)
    
    return train, test

class FaceDataset(Dataset):
    '''Class creates FaceDataset object
       Args:
        data_annotations (pandas.DataFrame) : contains image path and real/synthetic, gender, smile labels'''
        
    def __init__(self, data_annotations: pd.DataFrame, transform=None):
        self.img_labels = data_annotations
        
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):

        img_path = self.img_labels.iloc[idx, 0]
        
        image = read_image(img_path)
        
        if self.transform:
            image = self.transform(image)

        return image, self.img_labels.iloc[idx, 1], self.img_labels.iloc[idx, 2]

if __name__ == "__main__":
    
    df = pd.DataFrame(columns=["Sample", "Male", "Smiling"])
    df = load_files_from_dir("celeb", df)
    df.to_csv("dataset.csv")
    
    train, test = split_sets("dataset.csv")
    train, test = FaceDataset(train, img_transforms), FaceDataset(test, img_transforms)
    train, test = DataLoader(train, batch_size=2), DataLoader(test, batch_size=1)

    torch.save(train, 'train.pt'), torch.save(test, 'test.pt')
