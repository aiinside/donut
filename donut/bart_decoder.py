from typing import Any, List, Optional, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
from transformers import MBartConfig, MBartForCausalLM, AutoTokenizer, PreTrainedTokenizerFast, LiltModel
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.mbart.modeling_mbart import _expand_mask
from transformers.utils import logging

logger = logging.get_logger(__name__)

class PositionEmbbedings2d(nn.Module):
    def __init__(self, max_2d_position_embeddings=1024, hidden_size=1024, pad_token_id=0):
        super().__init__()
        # we divide the hidden_size by 6 here as there are 6 different layout embeddings,
        # namely left_position, upper_position, right_position, lower_position, height, width
        self.x_position_embeddings = nn.Embedding(max_2d_position_embeddings, hidden_size // 4, padding_idx=pad_token_id)
        self.y_position_embeddings = nn.Embedding(max_2d_position_embeddings, hidden_size // 4, padding_idx=pad_token_id)
        self.h_position_embeddings = nn.Embedding(max_2d_position_embeddings, hidden_size // 4, padding_idx=pad_token_id)
        self.w_position_embeddings = nn.Embedding(max_2d_position_embeddings, hidden_size // 4, padding_idx=pad_token_id)

        self.padding_idx = pad_token_id

    def forward(self, bbox=None, position_ids=None):
        try:
            center_x_embeddings = self.x_position_embeddings(bbox[:, :, 0])
            center_y_embeddings = self.y_position_embeddings(bbox[:, :, 1])
            width_embeddings = self.w_position_embeddings(bbox[:, :, 2])
            height_embeddings = self.h_position_embeddings(bbox[:, :, 3])
        except IndexError as e:
            raise IndexError("The `bbox` coordinate values should be within 0-1000 range.") from e

        spatial_position_embeddings = torch.cat(
            [
                center_x_embeddings,
                center_y_embeddings,
                height_embeddings,
                width_embeddings,
            ],
            dim=-1,
        )

        return spatial_position_embeddings

class BARTDecoder(nn.Module):
    """
    Donut Decoder based on Multilingual BART
    Set the initial weights and configuration with a pretrained multilingual BART model,
    and modify the detailed configurations as a Donut decoder

    Args:
        decoder_layer:
            Number of layers of BARTDecoder
        max_position_embeddings:
            The maximum sequence length to be trained
        name_or_path:
            Name of a pretrained model name either registered in huggingface.co. or saved in local,
            otherwise, `hyunwoongko/asian-bart-ecjk` will be set (using `transformers`)
    """

    def __init__(
        self, decoder_layer: int, max_position_embeddings: int, name_or_path: str = None,
    ):
        super().__init__()
        self.decoder_layer = decoder_layer
        self.max_position_embeddings = max_position_embeddings

        self.tokenizer = AutoTokenizer.from_pretrained(
            "rinna/bilingual-gpt-neox-4b-8k" if not name_or_path else name_or_path, use_fast=False
        )

        self.model = MBartForCausalLM(
            config=MBartConfig(
                is_decoder=True,
                is_encoder_decoder=False,
                add_cross_attention=True,
                decoder_layers=self.decoder_layer,
                max_position_embeddings=self.max_position_embeddings,
                vocab_size=len(self.tokenizer),
                scale_embedding=True,
                add_final_layer_norm=True,
                chunk_size_feed_forward=512
            )
        )
        self.model.forward = self.forward  # to get cross attentions and utilize `generate` function

        self.model.config.is_encoder_decoder = True  # to get cross-attention
        self.add_special_tokens(["<sep/>"])  # <sep/> is used for representing a list in a JSON
        self.model.model.decoder.embed_tokens.padding_idx = self.tokenizer.pad_token_id
        self.model.prepare_inputs_for_generation = self.prepare_inputs_for_inference

        self.pos_emb2d = PositionEmbbedings2d(max_2d_position_embeddings=max_position_embeddings)

    @property
    def decoder(self):
        return self.model.model.decoder
    
    @property
    def config(self):
        return self.model.config

    def add_special_tokens(self, list_of_tokens: List[str]):
        """
        Add special tokens to tokenizer and resize the token embeddings
        """
        newly_added_num = self.tokenizer.add_special_tokens({"additional_special_tokens": sorted(set(list_of_tokens))})
        if newly_added_num > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))

    def prepare_inputs_for_inference(self, input_ids: torch.Tensor, encoder_outputs: torch.Tensor, past=None, use_cache: bool = None, attention_mask: torch.Tensor = None):
        """
        Args:
            input_ids: (batch_size, sequence_lenth)
        Returns:
            input_ids: (batch_size, sequence_length)
            attention_mask: (batch_size, sequence_length)
            encoder_hidden_states: (batch_size, sequence_length, embedding_dim)
        """
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
        if past is not None:
            input_ids = input_ids[:, -1:]
        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past,
            "use_cache": use_cache,
            "encoder_hidden_states": encoder_outputs.last_hidden_state,
        }
        return output

    def forward_mbart(
        self,
        input_ids,
        boxes: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        past_key_values: Optional[torch.Tensor] = None,
        use_cache: bool = None,
        output_attentions: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[torch.Tensor] = None,
        return_dict: bool = None,
    ):
        # オリジナルのforwardから1Dのembを外して2Dにするために仕方なくほとんどコピー
        # 使わない引数へらした

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        input = input_ids
        input_shape = input.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        inputs_embeds = self.decoder.embed_tokens(input_ids) * self.decoder.embed_scale

        attention_mask = self.decoder._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # embed positions
        positions = self.pos_emb2d(boxes)

        hidden_states = inputs_embeds + positions.to(inputs_embeds.device)
        hidden_states = self.decoder.layernorm_embedding(hidden_states)

        hidden_states = nn.functional.dropout(hidden_states, p=self.decoder.dropout, training=self.training)

        if self.decoder.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing`. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.decoder.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.decoder.layerdrop:
                    continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.decoder.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    None,
                    None,
                    None,
                    past_key_value,
                    output_attentions,
                    use_cache
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=None,
                    layer_head_mask=None,
                    cross_attn_layer_head_mask=None,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        hidden_states = self.decoder.layer_norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )

    def forward(
        self,
        input_ids,
        boxes: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        past_key_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = None,
        output_attentions: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[torch.Tensor] = None,
        return_dict: bool = None,
    ):
        """
        A forward fucntion to get cross attentions and utilize `generate` function

        Source:
        https://github.com/huggingface/transformers/blob/v4.11.3/src/transformers/models/mbart/modeling_mbart.py#L1669-L1810

        Args:
            input_ids: (batch_size, sequence_length)
            attention_mask: (batch_size, sequence_length)
            encoder_hidden_states: (batch_size, sequence_length, hidden_size)

        Returns:
            loss: (1, )
            logits: (batch_size, sequence_length, hidden_dim)
            hidden_states: (batch_size, sequence_length, hidden_size)
            decoder_attentions: (batch_size, num_heads, sequence_length, sequence_length)
            cross_attentions: (batch_size, num_heads, sequence_length, sequence_length)
        """
        output_attentions = output_attentions if output_attentions is not None else self.model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.model.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.model.config.use_return_dict
        outputs = self.forward_mbart(
            input_ids=input_ids,
            boxes=boxes,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.model.lm_head(outputs[0])

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.model.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return ModelOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            decoder_attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
            decoder_hidden_states=outputs.hidden_states
        )
