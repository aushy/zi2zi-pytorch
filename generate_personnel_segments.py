from font2img import draw_single_char
from model import Zi2ZiModel

from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
import torch.nn as nn
import PIL
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

def generate_characters(
        src_txt, 
        infer_dir="./personnel_segments/pr/generated_characters", 
        experiment_dir="./experiment_pr", 
        src_font_path="./NotoSansCJKjp-Regular.otf",
        resume_iter=1000, 
        label=9
    ):

    # set up directories
    checkpoint_dir = os.path.join(experiment_dir, "checkpoint")
    os.makedirs(infer_dir, exist_ok=True)

    # model
    with torch.no_grad():

        # set up 
        model = Zi2ZiModel(
            input_nc=1,
            embedding_num=80,
            embedding_dim=128,
            Lconst_penalty=15,
            Lcategory_penalty=100,
            save_dir=checkpoint_dir,
            gpu_ids=["cuda:0"],
            g_norm_layer=nn.InstanceNorm2d,
            d_spec_norm=False,
            is_training=False
        )
        model.setup()
        model.print_networks(True)
        model.load_networks(resume_iter)

        # data prep
        src = src_txt
        font = ImageFont.truetype(src_font_path, size=256)
        img_list = [transforms.Normalize(0.5, 0.5)(transforms.ToTensor()(draw_single_char(ch, font, 256))).unsqueeze(dim=0) for ch in src]
        label_list = [label for _ in src]
        img_list = torch.cat(img_list, dim=0)
        label_list = torch.tensor(label_list)
        dataset = TensorDataset(label_list, img_list, img_list)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
        
        # inference
        for batch in dataloader:
            model.sample(batch, infer_dir)

if __name__ == '__main__':
    
    generate_characters("取締役")