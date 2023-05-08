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

from donut import DonutModel, DonutConfig
# import donut.model_custom
# from donut.model_custom import DonutModel, DonutConfig
from donut import DonutDataset
from lightning_module import DonutDataPLModule, DonutModelPLModule
import  glob, re
import tqdm

def img_glob(path):
    ext = r'.*\.(jpg|jpeg|png|tif|tiff|gif)$'
    path = os.path.join(path, '*')
    files = glob.glob(path)
    files = [f for f in files if re.search(ext, f, re.IGNORECASE)]
    return files

def test(config:Config, img_dir:str, save_path:str):
    pretrained_model = DonutModel.from_pretrained(
        config.pretrained_model_name_or_path,
        input_size=config.input_size,
        max_length=config.max_length,
        align_long_axis=config.align_long_axis,
    )

    if torch.cuda.is_available():
        pretrained_model.half()
        pretrained_model.to("cuda")

    pretrained_model.eval()

    if save_path:
        os.makedirs(save_path, exist_ok=True)

    img_list = sorted(img_glob(img_dir))

    for ii, img_path in tqdm.tqdm(enumerate(img_list)):
        image=PIL.Image.open(img_path)
        outs = pretrained_model.inference(image, prompt='<s_health>')
        # outs = pretrained_model.inference(image, prompt='<s_health>', return_attentions=True, return_confs=True, return_tokens=True)
        fn = os.path.basename(os.path.splitext(img_path)[0]) + '.json'
        fn = os.path.join(save_path, fn)
        print(outs['predictions'])
        with open(fn, 'w') as fp:
            json.dump(outs['predictions'], fp, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, default="test")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--save_path", type=str, default=None)
    args, left_argv = parser.parse_known_args()

    config = Config(args.config)

    test(config, args.img_dir, args.save_path)
