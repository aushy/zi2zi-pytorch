from font2img import draw_single_char
from model import Zi2ZiModel

from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
import torch.nn as nn
import PIL
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision.utils as vutils
import argparse
import numpy as np
import glob
import math


def generate_characters(
        src_txt, 
        infer_id,
        model,
        infer_dir="./personnel_segments/pr/generated_characters",
        src_font_path="./NotoSansCJKjp-Regular.otf",
        label=9
    ):

    # data prep
    src = src_txt
    font = ImageFont.truetype(src_font_path, size=256)
    img_list = [transforms.Normalize(0.5, 0.5)(
        transforms.ToTensor()(
            draw_single_char(ch, font, 256)
        )
    ).unsqueeze(dim=0) for ch in src]
    label_list = [label for _ in src]
    img_list = torch.cat(img_list, dim=0)
    label_list = torch.tensor(label_list)
    dataset = TensorDataset(label_list, img_list, img_list)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    # inference
    with torch.no_grad():
        cnt = 0
        for batch in dataloader:
            model.set_input(batch[0], batch[2], batch[1])
            model.forward()
            tensor_to_plot = model.fake_B
            for label, image_tensor in zip(batch[0], tensor_to_plot):
                vutils.save_image(
                    image_tensor, 
                    os.path.join(infer_dir, '_'.join([str(infer_id), str(cnt)]) + '.png')
                )
                cnt += 1


def generate_personnel_entries(positions_df, names_df, doc, sample_size=1):
    
    # position
    doc_positions_df = positions_df[positions_df['source']==doc]
    jp_positions = doc_positions_df['position_jp'].to_numpy()
    random_position = np.random.choice(jp_positions, size=sample_size, replace=True)
    
    # name
    doc_family_names_df = names_df[names_df['name_type']=='family']
    family_names = doc_family_names_df['Name'].to_numpy()
    
    doc_given_names_df = names_df[names_df['name_type']=='given']
    given_names = doc_given_names_df['Name'].to_numpy()
    
    random_family_name = np.random.choice(family_names, size=sample_size, replace=True)
    random_given_name = np.random.choice(given_names, size=sample_size, replace=True)

    return zip(random_position, random_family_name, random_given_name)


def image_concat(image_paths, resize, max_row_chars=4):

    # get images
    images = [Image.open(x).resize(resize, PIL.Image.ANTIALIAS) for x in image_paths]
    widths, heights = zip(*(i.size for i in images))

    # create block image
    block_width = min(len(images)*max(widths), max_row_chars*max(widths))
    total_rows = math.ceil(len(images) / max_row_chars)
    block_height = total_rows*max(heights)
    block_im = Image.new('RGB', (block_width, block_height), (255, 255, 255))

    # concatenate
    x_offset = 0
    y_offset = 0
    count = 0
    for im in images:
        if count % max_row_chars == 0 and count != 0:
            x_offset = 0
            y_offset += im.size[1]
        block_im.paste(im, (x_offset, y_offset))
        x_offset += im.size[0]
        count +=1
        
    return block_im


def generate_personnel_blocks(
        count, 
        positions_path, 
        names_path, 
        infer_dir,
        save_dir,
        small_char_size=(20,20),
        large_char_size=(30,30),
        personnel_entry_labels=("position", "family_name", "given_name"),
        resume_iter=1000,
        experiment_dir="./experiment_pr"
    ):

    # setup
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(infer_dir, exist_ok=True)
    checkpoint_dir = os.path.join(experiment_dir, "checkpoint")

    # read in personnel info
    personnel_positions_df = pd.read_csv(positions_path, header=0)
    personnel_names_df = pd.read_csv(names_path, header=0)

    # create random personnel entries
    personnel_entries = generate_personnel_entries(
        personnel_positions_df, 
        personnel_names_df, 
        doc="pr", sample_size=count
    )

    # create model
    model = Zi2ZiModel(
        input_nc=1,
        embedding_num=40,
        embedding_dim=128,
        Lconst_penalty=15,
        Lcategory_penalty=100,
        save_dir=checkpoint_dir,
        gpu_ids=["cuda:0"],
        g_norm_layer=nn.InstanceNorm2d,
        spec_norm=True,
        attention=True,
        is_training=False
    )
    model.setup()
    model.print_networks(True)
    model.load_networks(resume_iter)

    # generate blocks
    for person_id, person_chars in enumerate(personnel_entries):
        for chars_label, chars in zip(personnel_entry_labels, person_chars):

            # create id
            inference_id = '_'.join([str(person_id), chars_label])

            # generate chars for block
            generate_characters(
                chars, 
                model=model,
                infer_id=inference_id, 
                infer_dir=infer_dir
            )

            # get path info from gen chars
            infer_pattern = os.path.join(infer_dir, inference_id + '*')
            infer_char_paths = glob.glob(infer_pattern)

            # compose block image and save
            char_size = small_char_size if chars_label == "position" else large_char_size
            personnel_block_image = image_concat(infer_char_paths, char_size)
            personnel_block_image.save(os.path.join(save_dir, inference_id + ".png"))


if __name__ == '__main__':

    # TODO add stochastic variation

    parser = argparse.ArgumentParser(description='Block generation')
    parser.add_argument('--personnel_count', type=int, required=True, help="")
    parser.add_argument('--positions_csv', required=True, help="")
    parser.add_argument('--names_csv', required=True, help="")
    parser.add_argument('--inference_dir', required=True, help="")
    parser.add_argument('--save_dir', required=True, help="")
    parser.add_argument('--experiment_dir', required=True, help="")
    parser.add_argument('--checkpoint_iter', required=True, help="")
    args = parser.parse_args()

    generate_personnel_blocks(
        count=args.personnel_count, 
        positions_path=args.positions_csv, 
        names_path=args.names_csv, 
        infer_dir=args.inference_dir,
        save_dir=args.save_dir,
        experiment_dir=args.experiment_dir,
        resume_iter=args.checkpoint_iter
    )
