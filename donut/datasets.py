import json
import os
from dataclasses import dataclass
import re

import numpy as np
import PIL.Image
import torch
from torch.utils.data import Dataset
from transformers import T5Tokenizer

from .maxvit import prepare_input as prepare_img_tensor

def _is_unicode_tag(text):
    mm =  re.match('<0x[\dA-F][\dA-F]>', text)

    return mm != None

class FullOcrData():
    BOX_ATTRS = ['left', 'top', 'right', 'bottom']

    def __init__(self, json_data, imshape) -> None:
        self.imshape = imshape
        self.json_data = json_data
        pass

    def get_texts(self):
        return [result['text'] for result in self.json_data['results'] if len(result['text']) != 0]
    
    def get_char_boxes(self):
        box_lists = []
        scale = np.array([[self.imshape[0], self.imshape[1], self.imshape[0], self.imshape[1]],], dtype=np.float32)
        for result in self.json_data['results']:
            if len(result['text']) == 0:
                continue
            box_list = []
            for char in result['characters']:
                box = [char['bbox'][attr] for attr in self.BOX_ATTRS]
                box_list.append(box)

            box_list = np.array(box_list, dtype=np.float32)
            box_list = box_list * scale
            box_lists.append(box_list)

        return box_lists

class DonutDataset(Dataset):
    def __init__(self, image_list_path:str, tokenizer:T5Tokenizer,
                  max_token_len:int, max_position_len:int, img_size:int, drop_rate:float) -> None:
        super().__init__()

        self.image_list = self._read_img_list(image_list_path)
        self.max_token_len = max_token_len
        self.tokenizer = tokenizer
        self.max_position_len = max_position_len
        if isinstance(img_size, tuple) or isinstance(img_size, list):
            self.img_size = img_size
        else:
            self.img_size = (img_size, img_size)

        self.drop_rate = drop_rate

    def _read_img_list(self, image_list_path):
        outs = []
        with open(image_list_path) as fp:
            lines = fp.readlines()
            for rr in lines:
                rr = rr.strip("\n")
                if len(rr) <= 1:
                    continue
                assert os.path.exists(rr)
                outs.append(rr)
        return outs

    def __len__(self):
        return len(self.image_list)

    def _read_data(self, index):
        img_path = self.image_list[index]
        img = np.asarray(PIL.Image.open(img_path).convert('RGB'))
        json_path = os.path.splitext(img_path)[0] + '.json'
        with open(json_path) as fp:
            jdata = json.load(fp)
        focr = FullOcrData(jdata, (img.shape[1], img.shape[0]))

        return img, focr

    def _offsets_from_tokens(self, tokens, text):
        start = 0
        end = 0
        id_offsets = []
        for ii, tt in enumerate(tokens):

            if _is_unicode_tag(tt):
                if ii==0 or (not _is_unicode_tag(tokens[ii-1])):
                    start = end
                    end += 1
                id_offsets.append([start, end])
            else:
                start = end
                end += len(tt)
                id_offsets.append([start, end])

        return id_offsets

    def _get_token_boxes(self, char_box_lists, id_offsets_list):
        token_box_lists = []

        for char_boxes, id_offsets in zip(char_box_lists, id_offsets_list):
            token_boxes = []
            for (start, end) in id_offsets:
                targets = char_boxes[start:end]
                if len(targets) == 0:
                    targets = char_boxes[-1][np.newaxis]

                x0 = targets[:,0].min()
                y0 = targets[:,1].min()
                x1 = targets[:,2].max()
                y1 = targets[:,3].max()
                cx = (x0+x1)/2
                cy = (y0+y1)/2
                ww = x1-x0
                hh = y1-y0

                token_boxes.append([cx,cy,ww,hh])
            token_box_lists.append(token_boxes)

        return token_box_lists

    def prepare_textual_tensor(self, focr:FullOcrData):
        texts = focr.get_texts()
        char_box_lists = focr.get_char_boxes()

        input_ids_list = self.tokenizer.batch_encode_plus(texts, add_special_tokens=False)['input_ids']
        tokens_list = [self.tokenizer.convert_ids_to_tokens(ids) for ids in input_ids_list]
        id_offsets_list = [self._offsets_from_tokens(tokens, text) for tokens, text in zip(tokens_list, texts)]
        token_box_lists = self._get_token_boxes(char_box_lists, id_offsets_list)
        input_ids = sum(input_ids_list, [])
        token_boxes = sum(token_box_lists, [])

        if len(input_ids) != len(token_boxes):
            print(input_ids)
            print(token_boxes)
            print('====')

        pad_len = self.max_token_len - len(input_ids)
        if pad_len < 1:
            print(len(input_ids))
            input_ids = input_ids[:self.max_token_len]
            token_boxes = token_boxes[:self.max_token_len]
        else:
            input_ids += [self.tokenizer.pad_token_id] * pad_len
            token_boxes += [[0,0,0,0]] * pad_len

        return torch.Tensor(input_ids).long(), torch.Tensor(token_boxes).float()
        
    def __getitem__(self, index):
        img, focr = self._read_data(index)
        img_tensor, resize_rate = prepare_img_tensor(img, self.img_size)
        input_ids, token_boxes = self.prepare_textual_tensor(focr)

        mask = input_ids == self.tokenizer.pad_token_id
        box_scale = torch.Tensor([[resize_rate/self.img_size[0], resize_rate/self.img_size[1], 
                                   resize_rate/self.img_size[0], resize_rate/self.img_size[1]]]).float()
        token_boxes *= box_scale
        # 0 for pad, 1 for mask
        box_ids = (2 + token_boxes * (self.max_position_len-2)).long().clamp(0, self.max_position_len)
        box_ids[mask] = 0
        labels = input_ids.clone()
        labels[mask] = -100

        if 0 < self.drop_rate:
            mask = input_ids != self.tokenizer.pad_token_id
            elems = torch.count_nonzero(mask).item()
            drop_cands = torch.randint(0, elems, (max(1, int(elems*self.drop_rate)),))
            input_ids[mask][drop_cands] = self.tokenizer.mask_token_id
            box_ids[mask][drop_cands] = 1

        return img_tensor, input_ids, box_ids, labels, token_boxes
