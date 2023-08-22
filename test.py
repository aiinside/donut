"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
import argparse
import json
import os
from os.path import basename
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.plugins import CheckpointIO
from pytorch_lightning.utilities import rank_zero_only
from sconf import Config
import PIL.Image
from PIL import ImageFont, ImageDraw, Image
import cv2
import numpy as np
import unicodedata

from donut import DonutModel, DonutConfig
# import donut.model_custom
# from donut.model_custom import DonutModel, DonutConfig
from donut import DonutDataset
from lightning_module import DonutDataPLModule, DonutModelPLModule
import  glob, re
import tqdm
default_font = '/usr/share/fonts/truetype/fonts-japanese-mincho.ttf'

def draw_box(image, box, color, thickness=2):
    """ Draws a box on an image with a given color.

    # Arguments
        image     : The image to draw on.
        box       : A list of 4 elements (x1, y1, x2, y2).
        color     : The color of the box.
        thickness : The thickness of the lines to draw a box with.
    """
    b = np.array(box).astype(int)
    cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, thickness, cv2.LINE_AA)


def draw_captions(image, draw_info, back_white=False):
    font_size = max(16, int(image.shape[0] / 130))
    font = ImageFont.truetype(default_font, font_size)

    if back_white:
        for box, _, caption in draw_info:
            left, top = box[0], max(0, box[1] - font_size)
            caption_len = sum([2 if unicodedata.east_asian_width(c) in 'FWA' else 1 for c in caption])
            height = font_size
            width = int(caption_len * font_size * 0.6)
            image[top:top+height, left:left+width] = 255

    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    for box, color, caption in draw_info:
        left, top = box[0], max(0, box[1] - font_size)
        draw.text((left, top), caption, font=font, fill=color)

    image = np.array(img_pil)

    return image

def img_glob(path):
    ext = r'.*\.(jpg|jpeg|png|tif|tiff|gif)$'
    path = os.path.join(path, '*')
    files = glob.glob(path)
    files = [f for f in files if re.search(ext, f, re.IGNORECASE)]
    return files

def test(config:Config, img_dir:str, save_path:str, score_thresh):
    pretrained_model = DonutModel.from_pretrained(
        config.pretrained_model_name_or_path,
        input_size=config.input_size,
        max_length=config.max_length,
        align_long_axis=config.align_long_axis,
        enable_char_map=config.char_map,
        box_pred=config.get('box_pred',False)
    )

    if 'classes' in config:
        stokens = []
        for cn in config.classes:
            stokens += [fr"<s_{cn}>", fr"</s_{cn}>"]
        pretrained_model.decoder.add_special_tokens(stokens)

    if torch.cuda.is_available():
        pretrained_model.half()
        pretrained_model.to("cuda")

    pretrained_model.eval()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        viz_dir = os.path.join(save_path, 'viz')
        os.makedirs(viz_dir, exist_ok=True)

    img_list = sorted(img_glob(img_dir))

    for ii, img_path in tqdm.tqdm(enumerate(img_list)):
        image=PIL.Image.open(img_path)
        with torch.no_grad():
            outs = pretrained_model.inference_custom(image, prompt='<s_invoice_nttd>', token_score_thresh=score_thresh)
        # outs = pretrained_model.inference(image, prompt='<s_health>')
        # outs = pretrained_model.inference(image, prompt='<s_health>', return_attentions=True, return_confs=True, return_tokens=True)
        fn = os.path.basename(os.path.splitext(img_path)[0]) + '.json'
        fn = os.path.join(save_path, fn)
        print(img_path)
        
        # data = outs['predictions']
        if 'health' in outs['predictions']:
            data = outs['predictions']['health']['content']
        elif 'invoice' in outs['predictions']:
            data = outs['predictions']['invoice']['content']
        else:
            data = outs['predictions']

        with open(fn, 'w') as fp:
            json.dump(data, fp, ensure_ascii=False, indent=4)

        if not config.get('box_pred',False):
            continue

        image = np.asarray(image)
        hh, ww = image.shape[:2]
        for ind in data:
            if not isinstance(ind, dict):
                continue
            if 0 == len(ind):
                continue
            try:
                cname, content = list(ind.items())[0]
            except Exception as ex:
                print(ind)
                raise ex
            if not isinstance(content['content'], str):
                continue

            box = content['box']
            x0 = int(box[0] - box[2]//2)
            x0 = min(max(0, x0),ww)

            y0 = int(box[1] - box[3]//2)
            y0 = min(max(0, y0),hh)
            
            x1 = int(box[0] + box[2]//2)
            x1 = min(max(0, x1),ww)
            
            y1 = int(box[1] + box[3]//2)
            y1 = min(max(0, y1),hh)

            box = (x0,y0,x1,y1)
            msg = cname + ':' + content['content']

            draw_box(image, box, (0,255,0))
            image = draw_captions(image, [(box, (0,0,255), msg),])

        fn = os.path.basename(img_path)
        fn = os.path.join(viz_dir, fn)
        cv2.imwrite(fn, image[:,:,::-1])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, default="test")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--token_thresh", type=float, default=0.0)
    args, left_argv = parser.parse_known_args()

    config = Config(args.config)

    test(config, args.img_dir, args.save_path, args.token_thresh)
