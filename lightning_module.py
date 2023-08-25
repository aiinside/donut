"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
import math
import random
import re
from pathlib import Path

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from nltk import edit_distance
import Levenshtein
from pytorch_lightning.utilities import rank_zero_only
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from timm.optim import RAdam
from torch.distributed.optim.zero_redundancy_optimizer import ZeroRedundancyOptimizer


from donut import DonutConfig, DonutModel


class DonutModelPLModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        if self.config.get("pretrained_model_name_or_path", False):
            self.model = DonutModel.from_pretrained(
                self.config.pretrained_model_name_or_path,
                input_size=self.config.input_size,
                max_length=self.config.max_length,
                align_long_axis=self.config.align_long_axis,
                ignore_mismatched_sizes=True,
                swinv2=self.config.get('swinv2',False),
                enable_char_map=self.config.get('char_map', False),
                char_penalty = self.config.get('char_penalty', 2.0),
                box_pred = self.config.get('box_pred', False)
            )
        else:
            self.model = DonutModel(
                config=DonutConfig(
                    input_size=self.config.input_size,
                    max_length=self.config.max_length,
                    align_long_axis=self.config.align_long_axis,
                    swinv2=self.config.get('swinv2',False),
                    enable_char_map=self.config.get('char_map', False),
                    char_penalty = self.config.get('char_penalty', 2.0),
                    box_pred = self.config.get('box_pred', False)
                    # with DonutConfig, the architecture customization is available, e.g.,
                    # encoder_layer=[2,2,14,2], decoder_layer=4, ...
                )
            )

        if self.config.get('fix_encoder', False):
            for pp in self.model.encoder.parameters():
                pp.requires_grad = False

        self.valid_step_outs = []

    @property
    def enable_char_map(self):
        return self.config.get('char_map', False)

    def training_step(self, batch, batch_idx):
        image_tensors, decoder_input_ids, box_ids, decoder_labels, token_boxes = list(), list(), list(), list(), list()
        for batch_data in batch:
            image_tensors.append(batch_data[0])
            decoder_input_ids.append(batch_data[1])
            box_ids.append(batch_data[2])
            decoder_labels.append(batch_data[3])
            token_boxes.append(batch_data[4])

        image_tensors = torch.cat(image_tensors)
        decoder_input_ids = torch.cat(decoder_input_ids)
        box_ids = torch.cat(box_ids)
        decoder_labels = torch.cat(decoder_labels)
        token_boxes = torch.cat(token_boxes)
        loss = self.model(image_tensors, box_ids, decoder_input_ids, decoder_labels, box_labels=token_boxes)[0]
        self.log_dict({"train_loss": loss}, sync_dist=True)

        # image = image_tensors[0].detach().cpu().numpy()*255
        # image = image.transpose(1,2,0).copy().astype(np.uint8)
        # v1 = image.copy()
        # _boxes = box_ids[0].detach().cpu().numpy()/1024*image.shape[0]
        # for bb in _boxes.astype(np.int32):
        #     cx,cy,ww,hh = bb
        #     x0 = cx-ww//2
        #     x1 = cx+ww//2
        #     y0 = cy-hh//2
        #     y1 = cy+hh//2
        #     cv2.rectangle(image, (x0,y0), (x1,y1), (0,255,0), 2)
        # cv2.imwrite('hoge0.jpg', image)

        # _boxes = token_boxes[0].detach().cpu().numpy()
        # _boxes = _boxes*image.shape[0]
        # for bb in _boxes.astype(np.int32):
        #     cx,cy,ww,hh = bb
        #     x0 = cx-ww//2
        #     x1 = cx+ww//2
        #     y0 = cy-hh//2
        #     y1 = cy+hh//2
        #     cv2.rectangle(v1, (x0,y0), (x1,y1), (0,255,255), 2)

        # cv2.imwrite('hoge1.jpg', v1)

        # import pdb;pdb.set_trace()

        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        image_tensors, decoder_input_ids, box_ids, decoder_labels, token_boxes = list(), list(), list(), list(), list()
        for batch_data in batch:
            image_tensors.append(batch_data[0])
            decoder_input_ids.append(batch_data[1])
            box_ids.append(batch_data[2])
            decoder_labels.append(batch_data[3])
            token_boxes.append(batch_data[4])

        image_tensors = torch.cat(image_tensors)
        decoder_input_ids = torch.cat(decoder_input_ids)
        box_ids = torch.cat(box_ids)
        decoder_labels = torch.cat(decoder_labels)
        token_boxes = torch.cat(token_boxes)
        loss = self.model(image_tensors, box_ids, decoder_input_ids, decoder_labels, box_labels=token_boxes)[0]
        self.valid_step_outs.append(loss)

    def on_validation_epoch_end(self, validation_step_outputs):
        score = torch.stack(self.valid_step_outs).mean()
        self.log_dict({"val_metric": score}, sync_dist=True)
        self.valid_step_outs.clear()

    def configure_optimizers(self):

        max_iter = None

        if int(self.config.get("max_epochs", -1)) > 0:
            assert len(self.config.train_batch_sizes) == 1, "Set max_epochs only if the number of datasets is 1"
            max_iter = (self.config.max_epochs * self.config.num_training_samples_per_epoch) / (
                self.config.train_batch_sizes[0] * torch.cuda.device_count() * self.config.get("num_nodes", 1)
            )

        if int(self.config.get("max_steps", -1)) > 0:
            max_iter = min(self.config.max_steps, max_iter) if max_iter is not None else self.config.max_steps

        assert max_iter is not None
        optimizer = RAdam(self.parameters(), lr=self.config.lr)
        # optimizer = ZeroRedundancyOptimizer(self.parameters(), torch.optim.Adam, lr=self.config.lr)
        scheduler = {
            "scheduler": self.cosine_scheduler(optimizer, max_iter, self.config.warmup_steps),
            "name": "learning_rate",
            "interval": "step",
        }
        # return optimizer
        return [optimizer], [scheduler]

    @staticmethod
    def cosine_scheduler(optimizer, training_steps, warmup_steps):
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return current_step / max(1, warmup_steps)
            progress = current_step - warmup_steps
            progress /= max(1, training_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return LambdaLR(optimizer, lr_lambda)

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        items["exp_name"] = f"{self.config.get('exp_name', '')}"
        items["exp_version"] = f"{self.config.get('exp_version', '')}"
        return items

    @rank_zero_only
    def on_save_checkpoint(self, checkpoint):
        save_path = Path(self.config.result_path) / self.config.exp_name / self.config.exp_version
        self.model.save_pretrained(save_path)
        self.model.decoder.tokenizer.save_pretrained(save_path)


class DonutDataPLModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_batch_sizes = self.config.train_batch_sizes
        self.val_batch_sizes = self.config.val_batch_sizes
        self.train_datasets = []
        self.val_datasets = []
        self.g = torch.Generator()
        self.g.manual_seed(self.config.seed)

    def train_dataloader(self):
        loaders = list()
        for train_dataset, batch_size in zip(self.train_datasets, self.train_batch_sizes):
            loaders.append(
                DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    num_workers=self.config.num_workers,
                    pin_memory=True,
                    worker_init_fn=self.seed_worker,
                    generator=self.g,
                    shuffle=True,
                )
            )
        return loaders

    def val_dataloader(self):
        loaders = list()
        for val_dataset, batch_size in zip(self.val_datasets, self.val_batch_sizes):
            loaders.append(
                DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    pin_memory=True,
                    shuffle=False,
                )
            )
        return loaders

    @staticmethod
    def seed_worker(wordker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
