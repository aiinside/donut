"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
import json
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union

import torch
import zss
from nltk import edit_distance
from torch.utils.data import Dataset
from transformers.modeling_utils import PreTrainedModel
from zss import Node
import PIL.Image
import numpy as np
import cv2

from tqdm import tqdm

def decode_imtensor(tensor):

    from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    mean = torch.Tensor(IMAGENET_DEFAULT_MEAN).reshape(3,1,1)
    std = torch.Tensor(IMAGENET_DEFAULT_STD).reshape(3,1,1)
    img = ((tensor*std+mean)*255).permute(1,2,0).numpy().astype(np.uint8)
    return img

def save_json(write_path: Union[str, bytes, os.PathLike], save_obj: Any):
    with open(write_path, "w") as f:
        json.dump(save_obj, f)


def load_json(json_path: Union[str, bytes, os.PathLike]):
    with open(json_path, "r") as f:
        return json.load(f)

def load_dataset(dataset_name_or_path, split, dataset_root='/data/murayama/k8s/ocr_dxs1/donut/dataset', sort_key=False, classes:List[str]=None):
    if split == 'train':
        jname = 'train_data.json'
    else:
        jname = 'valid_data.json'

    mask_key = ['current_0', 'current_1', 'current_2', 'last_2_0', 'last_2_1', 'last_2_2', 'last_1_0', 'last_1_1', 'last_1_2']

    fn = os.path.join(dataset_root, dataset_name_or_path, jname)

    with open(fn) as fp:
        jdata = json.load(fp)
    
    def key_func(obj):
        return list(obj.keys())[0]

    def filter_entities(entities, classes):
        out = []
        for ent in entities:
            kk, vv = list(ent.items())[0]
            if (kk in classes): 
               out.append({kk:vv})
        return out

    def sort_by_key(entities):
        out = []
        for ent in sorted(entities, key=key_func):
            kk, vv = list(ent.items())[0]
            if kk in mask_key:
                continue
            out.append({kk:vv})
        return out

    for gt in jdata:
        _gt = gt['gt_parse']
        for ktask in _gt:
            entities = _gt[ktask]
            if classes:
                entities = filter_entities(entities, classes)

            if sort_key:
                entities = sort_by_key(entities)

            _gt[ktask] = entities

        if 'boxes' in gt:
            entities = gt['boxes']
            if classes:
                entities = filter_entities(entities, classes)
            if sort_key:
                entities = sort_by_key(entities)

            gt['boxes'] = entities
            
    return jdata

def resize_pad(img:np.ndarray, pad, target_size):
    th, tw = target_size
    px0,py0,_,_=pad

    out = np.zeros((th,tw), np.float32)
    ry, rx = th/img.shape[0], tw/img.shape[1]
    rr = min(ry, rx)
    hh = int(rr*img.shape[0])
    hh = min(hh, th-py0)
    ww = int(rr*img.shape[1])
    ww = min(ww, tw-px0)
    _img = cv2.resize(img, (ww, hh))
    out[py0:py0+_img.shape[0], px0:px0+_img.shape[1]] = _img

    return out

class DonutDataset(Dataset):
    """
    DonutDataset which is saved in huggingface datasets format. (see details in https://huggingface.co/docs/datasets)
    Each row, consists of image path(png/jpg/jpeg) and gt data (json/jsonl/txt),
    and it will be converted into input_tensor(vectorized image) and input_ids(tokenized string)

    Args:
        dataset_name_or_path: name of dataset (available at huggingface.co/datasets) or the path containing image files and metadata.jsonl
        ignore_id: ignore_index for torch.nn.CrossEntropyLoss
        task_start_token: the special token to be fed to the decoder to conduct the target task
    """

    def __init__(
        self,
        dataset_name_or_path: str,
        donut_model: PreTrainedModel,
        max_length: int,
        split: str = "train",
        ignore_id: int = -100,
        task_start_token: str = "<s>",
        prompt_end_token: str = None,
        sort_json_key: bool = True,
        classes:List[str] = None,
        return_charmap = False
    ):
        super().__init__()

        self.donut_model = donut_model
        self.max_length = max_length
        self.split = split
        self.ignore_id = ignore_id
        self.task_start_token = task_start_token
        self.prompt_end_token = prompt_end_token if prompt_end_token else task_start_token
        self.sort_json_key = sort_json_key
        self.return_charmap = return_charmap

        self.dataset = load_dataset(dataset_name_or_path, split=self.split, sort_key=self.sort_json_key, classes=classes)

        self.gt_token_sequences = []
        for sample in tqdm(self.dataset):
            # ground_truth = json.loads(sample["ground_truth"])
            ground_truth = sample
            if "gt_parses" in ground_truth:  # when multiple ground truths are available, e.g., docvqa
                assert isinstance(ground_truth["gt_parses"], list)
                gt_jsons = ground_truth["gt_parses"]
            else:
                assert "gt_parse" in ground_truth and isinstance(ground_truth["gt_parse"], dict)
                gt_jsons = [ground_truth["gt_parse"]]

            self.gt_token_sequences.append(
                [
                    task_start_token
                    + self.donut_model.json2token(
                        gt_json,
                        # update_special_tokens_for_json_key=self.split == "train",
                        update_special_tokens_for_json_key=False,
                        sort_json_key=self.sort_json_key,
                    )
                    + self.donut_model.decoder.tokenizer.eos_token
                    for gt_json in gt_jsons  # load json from list of json
                ]
            )
        keep = []
        for ii, sample in enumerate(tqdm(self.gt_token_sequences)):
            ids = self.donut_model.decoder.tokenizer(
                sample,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"].squeeze(0)
            if ids.shape[0] < self.max_length:
                keep.append(ii)

        if len(keep) != len(self.gt_token_sequences):
            print('long gt sequences are discarded {} -> {}'.format(len(self.gt_token_sequences), len(keep)))
            self.gt_token_sequences = [self.gt_token_sequences[ii] for ii in keep]
            self.dataset = [self.dataset[ii] for ii in keep]

        self.donut_model.decoder.add_special_tokens([self.task_start_token, self.prompt_end_token])
        self.prompt_end_token_id = self.donut_model.decoder.tokenizer.convert_tokens_to_ids(self.prompt_end_token)

    def __len__(self) -> int:
        return len(self.dataset)
    
    def prepare_bbox_target(self, boxes, target_tokens):
        vocab_size = self.donut_model.decoder.tokenizer.vocab_size + 1
        mask = target_tokens > vocab_size
        target_boxes = []
        cnt = 0
        pad = torch.zeros((4), dtype=torch.float32)
        pad[:] = -1
        for mm in mask:
            if mm:
                if (cnt//2) < len(boxes):
                    target_boxes.append(boxes[cnt//2])
                    cnt+=1
                else:
                    target_boxes.append(pad)
            else:
                target_boxes.append(pad)

        target_boxes = torch.stack(target_boxes)
        return target_boxes

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load image from image_path of given dataset_path and convert into input_tensor and labels.
        Convert gt data into input_ids (tokenized string)

        Returns:
            input_tensor : preprocessed image
            input_ids : tokenized gt_data
            labels : masked labels (model doesn't need to predict prompt and pad token)
        """
        sample = self.dataset[idx]
        image = PIL.Image.open(sample['image'])
        ww, hh = image.size

        # input_tensor
        input_tensor, pad = self.donut_model.encoder.prepare_input(image, random_padding=self.split == "train", return_padding=True)

        # input_ids
        processed_parse = random.choice(self.gt_token_sequences[idx])  # can be more than one, e.g., DocVQA Task 1
        input_ids = self.donut_model.decoder.tokenizer(
            processed_parse,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        # # char map
        fn = os.path.splitext(sample['image'])[0] + '.npy'
        cmap = np.load(fn)[:,:,0] # ch0:charmap, ch1:linkmap
        # cmap = np.load(fn).max(axis=-1)
        rate = cmap.shape[0] / max(hh, ww)
        ch, cw = int(hh*rate), int(ww*rate)
        cmap = cmap[:ch, :cw] # CRAFT予測時にpaddingした分を取る
        cmap = resize_pad(cmap, pad, self.donut_model.encoder.input_size)
        # cmap = cv2.resize(cmap, None, fx=1/4., fy=1/4., interpolation=cv2.INTER_AREA).astype(np.float32)
        hh, ww = self.donut_model.encoder.input_size
        # swinのwindowが4x4なので1/4サイズにする
        cmap = cmap.reshape(hh//4, 4, ww//4, 4).max(axis=(1,3))
        cmap = cmap[np.newaxis]

        ## bboxes
        if 'boxes' in sample:
            ww, hh = image.size
            boxes = [list(box.values())[0] for box in sample['boxes']]
            if len(boxes) == 0:
                boxes = torch.zeros((0,4), dtype=torch.float32)
            else:
                dw = pad[0] + pad[2]
                dh = pad[1] + pad[3]
                boxes = torch.Tensor(boxes)
                boxes[:,0::2] /= ww
                boxes[:,1::2] /= hh
                boxes[:,0::2] *= 1 - dw/self.donut_model.encoder.input_size[1]
                boxes[:,1::2] *= 1- dh/self.donut_model.encoder.input_size[0]
                
                boxes[:,0::2] += pad[0] / self.donut_model.encoder.input_size[1]
                boxes[:,1::2] += pad[1] / self.donut_model.encoder.input_size[0]
                # xyxy2cxcywh
                boxes[:,2] = boxes[:,2]-boxes[:,0]
                boxes[:,3] = boxes[:,3]-boxes[:,1]
                boxes[:,0] = boxes[:,0]+boxes[:,2]/2
                boxes[:,1] = boxes[:,1]+boxes[:,3]/2
        else:
            boxes = None

        if self.split == "train":
            labels = input_ids.clone()
            labels[
                labels == self.donut_model.decoder.tokenizer.pad_token_id
            ] = self.ignore_id  # model doesn't need to predict pad token
            labels[
                : torch.nonzero(labels == self.prompt_end_token_id).sum() + 1
            ] = self.ignore_id  # model doesn't need to predict prompt (for VQA)

            if not boxes is None:
                box_target = self.prepare_bbox_target(boxes, labels)
            else:
                box_target = None

            if not self.return_charmap:
                return input_tensor, input_ids, labels, box_target
            else:
                return input_tensor, input_ids, labels, box_target, cmap
        else:
            prompt_end_index = torch.nonzero(
                input_ids == self.prompt_end_token_id
            ).sum()  # return prompt end index instead of target output labels
            return input_tensor, input_ids, prompt_end_index, processed_parse


class JSONParseEvaluator:
    """
    Calculate n-TED(Normalized Tree Edit Distance) based accuracy and F1 accuracy score
    """

    @staticmethod
    def flatten(data: dict):
        """
        Convert Dictionary into Non-nested Dictionary
        Example:
            input(dict)
                {
                    "menu": [
                        {"name" : ["cake"], "count" : ["2"]},
                        {"name" : ["juice"], "count" : ["1"]},
                    ]
                }
            output(list)
                [
                    ("menu.name", "cake"),
                    ("menu.count", "2"),
                    ("menu.name", "juice"),
                    ("menu.count", "1"),
                ]
        """
        flatten_data = list()

        def _flatten(value, key=""):
            if type(value) is dict:
                for child_key, child_value in value.items():
                    _flatten(child_value, f"{key}.{child_key}" if key else child_key)
            elif type(value) is list:
                for value_item in value:
                    _flatten(value_item, key)
            else:
                flatten_data.append((key, value))

        _flatten(data)
        return flatten_data

    @staticmethod
    def update_cost(node1: Node, node2: Node):
        """
        Update cost for tree edit distance.
        If both are leaf node, calculate string edit distance between two labels (special token '<leaf>' will be ignored).
        If one of them is leaf node, cost is length of string in leaf node + 1.
        If neither are leaf node, cost is 0 if label1 is same with label2 othewise 1
        """
        label1 = node1.label
        label2 = node2.label
        label1_leaf = "<leaf>" in label1
        label2_leaf = "<leaf>" in label2
        if label1_leaf == True and label2_leaf == True:
            return edit_distance(label1.replace("<leaf>", ""), label2.replace("<leaf>", ""))
        elif label1_leaf == False and label2_leaf == True:
            return 1 + len(label2.replace("<leaf>", ""))
        elif label1_leaf == True and label2_leaf == False:
            return 1 + len(label1.replace("<leaf>", ""))
        else:
            return int(label1 != label2)

    @staticmethod
    def insert_and_remove_cost(node: Node):
        """
        Insert and remove cost for tree edit distance.
        If leaf node, cost is length of label name.
        Otherwise, 1
        """
        label = node.label
        if "<leaf>" in label:
            return len(label.replace("<leaf>", ""))
        else:
            return 1

    def normalize_dict(self, data: Union[Dict, List, Any]):
        """
        Sort by value, while iterate over element if data is list
        """
        if not data:
            return {}

        if isinstance(data, dict):
            new_data = dict()
            for key in sorted(data.keys(), key=lambda k: (len(k), k)):
                value = self.normalize_dict(data[key])
                if value:
                    if not isinstance(value, list):
                        value = [value]
                    new_data[key] = value

        elif isinstance(data, list):
            if all(isinstance(item, dict) for item in data):
                new_data = []
                for item in data:
                    item = self.normalize_dict(item)
                    if item:
                        new_data.append(item)
            else:
                new_data = [str(item).strip() for item in data if type(item) in {str, int, float} and str(item).strip()]
        else:
            new_data = [str(data).strip()]

        return new_data

    def cal_f1(self, preds: List[dict], answers: List[dict]):
        """
        Calculate global F1 accuracy score (field-level, micro-averaged) by counting all true positives, false negatives and false positives
        """
        total_tp, total_fn_or_fp = 0, 0
        for pred, answer in zip(preds, answers):
            pred, answer = self.flatten(self.normalize_dict(pred)), self.flatten(self.normalize_dict(answer))
            for field in pred:
                if field in answer:
                    total_tp += 1
                    answer.remove(field)
                else:
                    total_fn_or_fp += 1
            total_fn_or_fp += len(answer)
        return total_tp / (total_tp + total_fn_or_fp / 2)

    def construct_tree_from_dict(self, data: Union[Dict, List], node_name: str = None):
        """
        Convert Dictionary into Tree

        Example:
            input(dict)

                {
                    "menu": [
                        {"name" : ["cake"], "count" : ["2"]},
                        {"name" : ["juice"], "count" : ["1"]},
                    ]
                }

            output(tree)
                                     <root>
                                       |
                                     menu
                                    /    \
                             <subtree>  <subtree>
                            /      |     |      \
                         name    count  name    count
                        /         |     |         \
                  <leaf>cake  <leaf>2  <leaf>juice  <leaf>1
         """
        if node_name is None:
            node_name = "<root>"

        node = Node(node_name)

        if isinstance(data, dict):
            for key, value in data.items():
                kid_node = self.construct_tree_from_dict(value, key)
                node.addkid(kid_node)
        elif isinstance(data, list):
            if all(isinstance(item, dict) for item in data):
                for item in data:
                    kid_node = self.construct_tree_from_dict(
                        item,
                        "<subtree>",
                    )
                    node.addkid(kid_node)
            else:
                for item in data:
                    node.addkid(Node(f"<leaf>{item}"))
        else:
            raise Exception(data, node_name)
        return node

    def cal_acc(self, pred: dict, answer: dict):
        """
        Calculate normalized tree edit distance(nTED) based accuracy.
        1) Construct tree from dict,
        2) Get tree distance with insert/remove/update cost,
        3) Divide distance with GT tree size (i.e., nTED),
        4) Calculate nTED based accuracy. (= max(1 - nTED, 0 ).
        """
        pred = self.construct_tree_from_dict(self.normalize_dict(pred))
        answer = self.construct_tree_from_dict(self.normalize_dict(answer))
        return max(
            0,
            1
            - (
                zss.distance(
                    pred,
                    answer,
                    get_children=zss.Node.get_children,
                    insert_cost=self.insert_and_remove_cost,
                    remove_cost=self.insert_and_remove_cost,
                    update_cost=self.update_cost,
                    return_operations=False,
                )
                / zss.distance(
                    self.construct_tree_from_dict(self.normalize_dict({})),
                    answer,
                    get_children=zss.Node.get_children,
                    insert_cost=self.insert_and_remove_cost,
                    remove_cost=self.insert_and_remove_cost,
                    update_cost=self.update_cost,
                    return_operations=False,
                )
            ),
        )
