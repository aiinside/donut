"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
import math
import os
import re
from typing import Any, List, Optional, Union

import numpy as np
import PIL
from PIL.Image import Resampling
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import ImageOps
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.swin_transformer import SwinTransformer
from timm.models.swin_transformer_v2 import SwinTransformerV2
from torchvision import transforms
from torchvision.transforms.functional import resize, rotate, InterpolationMode
from torchvision.ops import complete_box_iou_loss, box_iou
from transformers import MBartConfig, MBartForCausalLM, AutoTokenizer, PreTrainedTokenizerFast, LiltModel
from transformers.file_utils import ModelOutput
from transformers.modeling_utils import PretrainedConfig, PreTrainedModel
from transformers.generation import LogitsProcessorList

from .maxvit import MaxxVitBase256
from .bart_decoder import BARTDecoder

def cxcywh_xyxy(box:torch.Tensor):
    xc,yc,ww,hh = box.unbind(-1)
    xyxy = [(xc-0.5*ww), (yc-0.5*hh), (xc+0.5*ww), (yc+0.5*hh)]
    return torch.stack(xyxy, dim=-1)

class DonutConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DonutModel`]. It is used to
    instantiate a Donut model according to the specified arguments, defining the model architecture

    Args:
        input_size:
            Input image size (canvas size) of Donut.encoder, SwinTransformer in this codebase
        align_long_axis:
            Whether to rotate image if height is greater than width
        window_size:
            Window size of Donut.encoder, SwinTransformer in this codebase
        encoder_layer:
            Depth of each Donut.encoder Encoder layer, SwinTransformer in this codebase
        decoder_layer:
            Number of hidden layers in the Donut.decoder, such as BART
        max_position_embeddings
            Trained max position embeddings in the Donut decoder,
            if not specified, it will have same value with max_length
        max_length:
            Max position embeddings(=maximum sequence length) you want to train
        name_or_path:
            Name of a pretrained model name either registered in huggingface.co. or saved in local
    """

    model_type = "donut"

    def __init__(
        self,
        input_size: List[int] = [2560, 1920],
        align_long_axis: bool = False,
        window_size: int = 10,
        encoder_layer: List[int] = [2, 2, 14, 2],
        decoder_layer: int = 4,
        max_position_embeddings: int = None,
        max_length: int = 1536,
        name_or_path: Union[str, bytes, os.PathLike] = "",
        enable_token_weight: bool = False,
        swinv2: bool = False,
        enable_char_map: bool = False,
        char_penalty: float = 2.,
        box_pred: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.align_long_axis = align_long_axis
        self.window_size = window_size
        self.encoder_layer = encoder_layer
        self.decoder_layer = decoder_layer
        self.max_position_embeddings = max_length if max_position_embeddings is None else max_position_embeddings
        self.max_length = max_length
        self.name_or_path = name_or_path
        self.enable_token_weight = enable_token_weight
        self.enable_char_map = enable_char_map
        self.swinv2 = swinv2
        self.char_penalty = char_penalty
        self.box_pred = box_pred

class DonutModel(PreTrainedModel):
    r"""
    Donut: an E2E OCR-free Document Understanding Transformer.
    The encoder maps an input document image into a set of embeddings,
    the decoder predicts a desired token sequence, that can be converted to a structured format,
    given a prompt and the encoder output embeddings
    """
    config_class = DonutConfig
    base_model_prefix = "donut"
    PRETRAINED_MAXXVIT = '/data/murayama/k8s/ocr_dxs1/donut/pretrained_weights/pretrained_maxxvit.pth'

    def __init__(self, config: DonutConfig):
        super().__init__(config)
        self.config = config
        self.encoder = MaxxVitBase256(use_fpn=True)

        self.decoder = BARTDecoder(
            max_position_embeddings=self.config.max_position_embeddings,
            decoder_layer=self.config.decoder_layer,
            name_or_path=self.config.name_or_path,
        )

        self.image_proj = nn.Sequential(
            nn.Linear(self.encoder.feature_channels[-2], 1024),
            nn.LayerNorm(1024)
        )

        self.box_head = nn.Sequential(
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, 5) # cxcywh+iou
        )

        if not config.name_or_path:
            state = torch.load(self.PRETRAINED_MAXXVIT, map_location='cpu')
            self.encoder.load_state_dict(state)

        self.encoder.freeze()
 
    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            module.bias.data.fill_(0)
            nn.init.xavier_uniform(module.weight)

    def forward_box_head(self, last_hidden_state: torch.Tensor, box_labels: torch.Tensor=None):
        if self.box_head:
            pred = self.box_head(last_hidden_state).sigmoid()
            box_pred = pred[:,:,:4]
            conf = pred[:,:,4]
            loss = None
            if not box_labels is None:
                mask = (box_labels > 0).all(dim=-1)
                if mask.sum() == 0:
                    loss = 0
                else:
                    # box数で割りたいので座標方向分掛ける
                    loss_l1 = F.l1_loss(box_pred[mask], box_labels[mask]) * 4

                    # iou loss
                    pred_xyxy = cxcywh_xyxy(box_pred)
                    label_xyxy = cxcywh_xyxy(box_labels)
                    loss_iou = complete_box_iou_loss(pred_xyxy[mask], label_xyxy[mask], reduction='mean')

                    # iou pred loss
                    # boxの尤度が知りたいのでiouを推定させてみる(samでもやってる)
                    _conf = conf[mask]
                    ious = box_iou(pred_xyxy[mask], label_xyxy[mask])
                    ious = torch.diagonal(ious, dim1=-2, dim2=-1)
                    loss_conf = F.l1_loss(_conf, ious)

                    # detrが cls:l1:giou = 2:5:2 だったので2.5倍しておく
                    loss = 2.5 * loss_l1 + loss_iou + loss_conf

            return loss, box_pred, conf
        
        return None, None, None

    def forward(self, image_tensors: torch.Tensor, box_ids: torch.Tensor, decoder_input_ids: torch.Tensor, decoder_labels: torch.Tensor, box_labels:torch.Tensor=None):
        """
        Calculate a loss given an input image and a desired token sequence,
        the model will be trained in a teacher-forcing manner

        Args:
            image_tensors: (batch_size, num_channels, height, width)
            decoder_input_ids: (batch_size, sequence_length, embedding_dim)
            decode_labels: (batch_size, sequence_length)
        """
        encoder_outputs = self.encoder(image_tensors)[-2]
        bs, ch, hh, ww = encoder_outputs.shape
        encoder_outputs = encoder_outputs.reshape(bs, ch, hh*ww).permute(0,2,1)
        encoder_outputs = self.image_proj(encoder_outputs)

        need_loss = decoder_labels != None
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            boxes = box_ids,
            encoder_hidden_states=encoder_outputs,
            labels=decoder_labels,
            output_attentions=need_loss,
            output_hidden_states=True
        )

        if box_labels is not None:
            box_loss, _, _ = self.forward_box_head(decoder_outputs[3][-1], box_labels)
            loss = decoder_outputs[0] + box_loss
            decoder_outputs = (loss,) + decoder_outputs[1:]

        return decoder_outputs

    def inference(
        self,
        image: PIL.Image = None,
        prompt: str = None,
        image_tensors: Optional[torch.Tensor] = None,
        prompt_tensors: Optional[torch.Tensor] = None,
        return_json: bool = True,
        return_attentions: bool = False,
        return_segmap: bool = False,
    ):
        """
        Generate a token sequence in an auto-regressive manner,
        the generated token sequence is convereted into an ordered JSON format

        Args:
            image: input document image (PIL.Image)
            prompt: task prompt (string) to guide Donut Decoder generation
            image_tensors: (1, num_channels, height, width)
                convert prompt to tensor if image_tensor is not fed
            prompt_tensors: (1, sequence_length)
                convert image to tensor if prompt_tensor is not fed
        """
        # prepare backbone inputs (image and prompt)
        if image is None and image_tensors is None:
            raise ValueError("Expected either image or image_tensors")
        if all(v is None for v in {prompt, prompt_tensors}):
            raise ValueError("Expected either prompt or prompt_tensors")

        if image_tensors is None:
            image_tensors = self.encoder.prepare_input(image).unsqueeze(0)

        if self.device.type == "cuda":  # half is not compatible in cpu implementation.
            image_tensors = image_tensors.half()
            image_tensors = image_tensors.to(self.device)

        if prompt_tensors is None:
            prompt_tensors = self.decoder.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]

        prompt_tensors = prompt_tensors.to(self.device)

        last_hidden_state = self.encoder(image_tensors)
        if self.device.type != "cuda":
            last_hidden_state = last_hidden_state.to(torch.float32)

        encoder_outputs = ModelOutput(last_hidden_state=last_hidden_state, attentions=None)

        if len(encoder_outputs.last_hidden_state.size()) == 1:
            encoder_outputs.last_hidden_state = encoder_outputs.last_hidden_state.unsqueeze(0)
        if len(prompt_tensors.size()) == 1:
            prompt_tensors = prompt_tensors.unsqueeze(0)

        # get decoder output
        decoder_output = self.decoder.model.generate(
            decoder_input_ids=prompt_tensors,
            encoder_outputs=encoder_outputs,
            max_length=self.config.max_length,
            early_stopping=True,
            pad_token_id=self.decoder.tokenizer.pad_token_id,
            eos_token_id=self.decoder.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[self.decoder.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
            output_attentions=return_attentions,
        )

        output = {"predictions": list()}
        for seq in self.decoder.tokenizer.batch_decode(decoder_output.sequences):
            seq = seq.replace(self.decoder.tokenizer.eos_token, "").replace(self.decoder.tokenizer.pad_token, "")
            seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token
            if return_json:
                output["predictions"].append(self.token2json(seq))
            else:
                output["predictions"].append(seq)

        if return_attentions:
            output["attentions"] = {
                "self_attentions": decoder_output.decoder_attentions,
                "cross_attentions": decoder_output.cross_attentions,
            }

        if return_segmap and self.config.enable_char_map:
            seg_map = self.forward_seg_head(last_hidden_state).sigmoid()
            output['segmap'] = seg_map

        return output
    

    def inference_custom(
        self,
        image: PIL.Image = None,
        prompt: str = None,
        image_tensors: Optional[torch.Tensor] = None,
        prompt_tensors: Optional[torch.Tensor] = None,
        return_json: bool = True,
        return_attentions: bool = False,
        return_segmap: bool = False,
        token_score_thresh = 0.0,
    ):
        """
        Generate a token sequence in an auto-regressive manner,
        the generated token sequence is convereted into an ordered JSON format

        Args:
            image: input document image (PIL.Image)
            prompt: task prompt (string) to guide Donut Decoder generation
            image_tensors: (1, num_channels, height, width)
                convert prompt to tensor if image_tensor is not fed
            prompt_tensors: (1, sequence_length)
                convert image to tensor if prompt_tensor is not fed
        """
        # prepare backbone inputs (image and prompt)
        if image is None and image_tensors is None:
            raise ValueError("Expected either image or image_tensors")
        if all(v is None for v in {prompt, prompt_tensors}):
            raise ValueError("Expected either prompt or prompt_tensors")

        if image_tensors is None:
            image_tensors, pad = self.encoder.prepare_input(image, return_padding=True)
            image_tensors = image_tensors.unsqueeze(0)
            dw = pad[0]+pad[2]
            dh = pad[1]+pad[3]
            ww, hh = image.size
            rx = ww / (1-dw/self.encoder.input_size[1])
            ry = hh / (1-dh/self.encoder.input_size[0])
            x0 = pad[0] / self.encoder.input_size[1]
            y0 = pad[1] / self.encoder.input_size[0]
        else:
            rx = ry = 1.0
            x0 = y0 = 0

        if self.device.type == "cuda":  # half is not compatible in cpu implementation.
            image_tensors = image_tensors.half()
            image_tensors = image_tensors.to(self.device)

        if prompt_tensors is None:
            prompt_tensors = self.decoder.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]

        prompt_tensors = prompt_tensors.to(self.device)

        last_hidden_state = self.encoder(image_tensors)
        if self.device.type != "cuda":
            last_hidden_state = last_hidden_state.to(torch.float32)

        encoder_outputs = ModelOutput(last_hidden_state=last_hidden_state, attentions=None)

        if len(encoder_outputs.last_hidden_state.size()) == 1:
            encoder_outputs.last_hidden_state = encoder_outputs.last_hidden_state.unsqueeze(0)
        if len(prompt_tensors.size()) == 1:
            prompt_tensors = prompt_tensors.unsqueeze(0)

        # そもそも予測の段階でthresholdより低いtokenを出さなくする
        sep_id = self.decoder.tokenizer.encode('<sep/>')[1]
        def skip_lowconf_token(input_ids: torch.LongTensor, logits: torch.FloatTensor):
            scores = logits.softmax(dim=-1)
            max_conf = scores.max(dim=-1)[0]
            mask = max_conf < token_score_thresh

            for ii, mm in enumerate(mask):
                if mm:
                    scores[ii] = 0.
                    scores[ii][sep_id] = 1.
                else:
                    scores[ii] = logits[ii]
            return scores

        logits_processor = LogitsProcessorList()
        logits_processor.append(skip_lowconf_token)

        # get decoder output
        decoder_output = self.decoder.model.generate(
            decoder_input_ids=prompt_tensors,
            encoder_outputs=encoder_outputs,
            max_length=self.config.max_length,
            early_stopping=True,
            pad_token_id=self.decoder.tokenizer.pad_token_id,
            eos_token_id=self.decoder.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[self.decoder.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
            output_attentions=return_attentions,
            output_scores=True,
            output_hidden_states=True,
            logits_processor=logits_processor
        )

        output = {"predictions": list()}

        # seq,nhead,1,1,1024, 最終層のattentionしか使わないので最後だけ抜き出す
        last_decoder_state = [ds[-1] for ds in decoder_output.decoder_hidden_states]
        last_decoder_state = torch.cat(last_decoder_state, dim=1)

        if self.box_head:
            _, boxes, box_scores = self.forward_box_head(last_decoder_state)
            boxes[:,:,0] -= x0
            boxes[:,:,1] -= y0
            boxes[:,:,0::2] *= rx
            boxes[:,:,1::2] *= ry
        else:
            bs, nseq = last_decoder_state.shape[:2]
            boxes = torch.zeros((bs, nseq, 4))
            box_scores = torch.zeros((bs, nseq))

        output['boxes'] = (boxes[0], box_scores[0])
        boxes = torch.cat((boxes, box_scores.unsqueeze(-1)), dim=-1)
        boxes = boxes.detach().cpu().numpy().tolist()
        scores = torch.amax(torch.cat(decoder_output.scores, dim=0).softmax(dim=-1), dim=-1)
        scores = scores.detach().cpu().numpy().tolist()
        predictions = self._decode(decoder_output.sequences[0,1:], scores, boxes[0], return_json)
        output["predictions"] = predictions

        len_prompt = prompt_tensors.shape[1]
        # 1画像入力でbatch_size=1とわかっているのでバラしておく
        output['token_ids'] = decoder_output.sequences[0, len_prompt:]
        output['tokens_scores'] = decoder_output.scores[0]


        if return_attentions:
            output["attentions"] = {
                "self_attentions": decoder_output.decoder_attentions,
                "cross_attentions": decoder_output.cross_attentions,
            }

        if return_segmap and self.config.enable_char_map:
            seg_map = self.forward_seg_head(last_hidden_state).sigmoid()
            output['segmap'] = seg_map

        return output

    def _decode(self, token_ids, scores, boxes, return_json=False):
        token_strs = self.decoder.tokenizer.convert_ids_to_tokens(token_ids)
        if return_json:
            return self.token2json_withbox(token_strs, scores, boxes)
        else:
            return self.decoder.tokenizer.convert_tokens_to_string(token_strs)

    def token2json_withbox(self, tokens, scores, boxes, *, is_inner_value=False):
        """
        Convert a (generated) token seuqnce into an ordered JSON format
        """
        output = dict()

        while len(tokens) != 0:
            start_token = None
            for ii, token in enumerate(tokens):
                start_token = re.search(r"<s_(.*?)>", token, re.IGNORECASE)
                if start_token is None:
                    continue
                istart = ii
                key = start_token.group(1)
                start_token = start_token.group()
                break

            if start_token is None:
                break

            end_token = None
            for ii in range(istart+1, len(tokens)):
                end_token = re.search(fr"</s_{key}>", tokens[ii], re.IGNORECASE)
                if end_token is None:
                    continue
                iend = ii
                end_token = end_token.group()
                break

            if end_token is None:
                tokens = tokens[istart+1:]
                boxes = boxes[istart+1:]
                scores = scores[istart+1:]
                continue

            contents = tokens[istart+1:iend]
            cboxes = boxes[istart+1:iend]
            cscores = scores[istart+1:iend]
            if 0 < len(contents):
                sfound = False
                efound = False
                output[key] = {}
                for cc in contents:
                    if "<s_" in cc:
                        sfound = True
                    if "</s_" in cc:
                        efound = True

                if sfound and efound:
                    values = self.token2json_withbox(contents, cscores, cboxes, is_inner_value=True)
                    if values:
                        if len(values) == 1:
                            values = values[0]
                        output[key]['content'] = values
                    else:
                        output[key]['content'] = ""
                else:
                    values = []
                    contents = self.decoder.tokenizer.convert_tokens_to_string(contents).strip()
                    for leaf in contents.split(r"<sep/>"):
                        leaf = leaf.strip()
                        if (
                            leaf in self.decoder.tokenizer.get_added_vocab()
                            and leaf[0] == "<"
                            and leaf[-2:] == "/>"
                        ):
                            leaf = leaf[1:-2]  # for categorical special tokens
                        values.append(leaf)

                    # if len(values) == 1:
                    #     values = values[0]
                    values = ''.join(values)
                    output[key]["content"] = values
            else:
                output[key] = {'content':''}

            score = min(scores[istart], scores[iend])
            start_box = boxes[istart]
            end_box = boxes[iend]
            box = end_box if start_box[4] < end_box[4] else start_box
            output[key]["box"] = box
            output[key]["score"] = score

            tokens = tokens[iend+1:]
            boxes = boxes[iend+1:]
            scores = scores[iend+1:]
            if len(tokens) != 0 and tokens[0] == '<sep/>':
                return [output] + self.token2json_withbox(tokens[1:], scores[1:], boxes[1:], is_inner_value=True)
        
        if len(output):
            return [output] if is_inner_value else output
        else:
            return [] if is_inner_value else {"text_sequence": tokens}

    def json2token(self, obj: Any, update_special_tokens_for_json_key: bool = True, sort_json_key: bool = True, can_value_key=False, out_sep=False):
        """
        Convert an ordered JSON object into a token sequence
        """
        if type(obj) == dict:
            if len(obj) == 1 and "text_sequence" in obj:
                return obj["text_sequence"]
            else:
                output = ""
                if sort_json_key:
                    keys = sorted(obj.keys(), reverse=True)
                else:
                    keys = obj.keys()
                for k in keys:
                    if update_special_tokens_for_json_key:
                        self.decoder.add_special_tokens([fr"<s_{k}>", fr"</s_{k}>"])
                    vv = obj[k]
                    if not out_sep or vv != '':
                        output += (
                            fr"<s_{k}>"
                            + self.json2token(obj[k], update_special_tokens_for_json_key, sort_json_key, can_value_key=can_value_key, out_sep=out_sep)
                            + fr"</s_{k}>"
                        )
                    else:
                        output += '<sep/>'

                return output
        elif type(obj) == list:
            return r"<sep/>".join(
                [self.json2token(item, update_special_tokens_for_json_key, sort_json_key, can_value_key=can_value_key, out_sep=out_sep) for item in obj]
            )
        else:
            obj = str(obj)
            if can_value_key:
                if f"<{obj}/>" in self.decoder.tokenizer.all_special_tokens:
                    obj = f"<{obj}/>"  # for categorical special tokens
            return obj

    def token2json(self, tokens, is_inner_value=False):
        """
        Convert a (generated) token seuqnce into an ordered JSON format
        """
        output = dict()

        while tokens:
            start_token = re.search(r"<s_(.*?)>", tokens, re.IGNORECASE)
            if start_token is None:
                break
            key = start_token.group(1)
            end_token = re.search(fr"</s_{key}>", tokens, re.IGNORECASE)
            start_token = start_token.group()
            if end_token is None:
                tokens = tokens.replace(start_token, "")
            else:
                end_token = end_token.group()
                start_token_escaped = re.escape(start_token)
                end_token_escaped = re.escape(end_token)
                content = re.search(f"{start_token_escaped}(.*?){end_token_escaped}", tokens, re.IGNORECASE)
                if content is not None:
                    content = content.group(1).strip()
                    if r"<s_" in content and r"</s_" in content:  # non-leaf node
                        value = self.token2json(content, is_inner_value=True)
                        if value:
                            if len(value) == 1:
                                value = value[0]
                            output[key] = value
                    else:  # leaf nodes
                        output[key] = []
                        for leaf in content.split(r"<sep/>"):
                            leaf = leaf.strip()
                            if (
                                leaf in self.decoder.tokenizer.get_added_vocab()
                                and leaf[0] == "<"
                                and leaf[-2:] == "/>"
                            ):
                                leaf = leaf[1:-2]  # for categorical special tokens
                            output[key].append(leaf)
                        if len(output[key]) == 1:
                            output[key] = output[key][0]

                tokens = tokens[tokens.find(end_token) + len(end_token) :].strip()
                if tokens[:6] == r"<sep/>":  # non-leaf nodes
                    return [output] + self.token2json(tokens[6:], is_inner_value=True)

        if len(output):
            return [output] if is_inner_value else output
        else:
            return [] if is_inner_value else {"text_sequence": tokens}

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, bytes, os.PathLike],
        *model_args,
        **kwargs,
    ):
        r"""
        Instantiate a pretrained donut model from a pre-trained model configuration

        Args:
            pretrained_model_name_or_path:
                Name of a pretrained model name either registered in huggingface.co. or saved in local,
                e.g., `naver-clova-ix/donut-base`, or `naver-clova-ix/donut-base-finetuned-rvlcdip`
        """
        model = super(DonutModel, cls).from_pretrained(pretrained_model_name_or_path, revision="official", *model_args, **kwargs)

        # truncate or interplolate position embeddings of donut decoder
        # max_length = kwargs.get("max_length", model.config.max_position_embeddings)
        # if (
        #     max_length != model.config.max_position_embeddings
        # ):  # if max_length of trained model differs max_length you want to train
        #     model.decoder.model.model.decoder.embed_positions.weight = torch.nn.Parameter(
        #         model.decoder.resize_bart_abs_pos_emb(
        #             model.decoder.model.model.decoder.embed_positions.weight,
        #             max_length
        #             + 2,  # https://github.com/huggingface/transformers/blob/v4.11.3/src/transformers/models/mbart/modeling_mbart.py#L118-L119
        #         )
        #     )
        #     model.config.max_position_embeddings = max_length

        return model
