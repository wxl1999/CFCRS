import math
import os
from typing import Tuple, Optional, Union, List, Dict, Any

import torch
from loguru import logger
from torch import nn
from torch.nn import functional as F, CrossEntropyLoss
from torch_geometric.nn import RGCNConv
from transformers import BartPretrainedModel, BartConfig, BartModel
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.utils import ModelOutput

from utils import SelfAttention, shift_tokens_right


class TokenEmbedding(nn.Module):
    def __init__(self, hidden_size, num_relations, num_bases, num_entities, num_special_tokens):
        super(TokenEmbedding, self).__init__()

        self.kg_encoder = RGCNConv(
            hidden_size, hidden_size, num_relations=num_relations, num_bases=num_bases
        )
        self.node_embeds = nn.Parameter(torch.empty(num_entities, hidden_size))
        stdv = math.sqrt(6.0 / (self.node_embeds.shape[-2] + self.node_embeds.shape[-1]))
        self.node_embeds.data.uniform_(-stdv, stdv)

        self.special_token_embeddings = nn.Embedding(num_special_tokens, hidden_size, padding_idx=0).weight

    def forward(self, edge_index, edge_type):
        node_embeddings = self.kg_encoder(self.node_embeds, edge_index, edge_type)
        token_embeddings = torch.cat([node_embeddings, self.special_token_embeddings], dim=0)
        return token_embeddings

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'model.pt')
        torch.save(self.state_dict(), save_path)

    def load(self, load_dir):
        load_path = os.path.join(load_dir, 'model.pt')
        missing_keys, unexpected_keys = self.load_state_dict(torch.load(load_path, map_location=torch.device('cpu')))
        logger.info(f"missing_keys: {missing_keys}, unexpected_keys: {unexpected_keys}")


class UserEncoder(nn.Module):
    def __init__(self, hidden_size):
        super(UserEncoder, self).__init__()
        self.self_attn = SelfAttention(hidden_size)

    def forward(self, user_entity_embeds=None, token_embeds=None, user_entity_ids=None, user_entity_mask=None):
        """
        Args:
            user_entity_embeds (batch_size, entity_len, hidden_size)
            token_embeds (num_entities, hidden_size)
            user_entity_ids (batch_size, entity_len)
            user_entity_mask (batch_size, entity_len)

        Returns:
            user_embeds (batch_size, hidden_size)
        """
        if user_entity_embeds is None:
            user_entity_embeds = token_embeds[user_entity_ids]  # (bs, ent_len, hs)

        user_embeds = self.self_attn(user_entity_embeds, user_entity_mask)
        return user_embeds

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'model.pt')
        torch.save(self.state_dict(), save_path)

    def load(self, load_dir):
        load_path = os.path.join(load_dir, 'model.pt')
        missing_keys, unexpected_keys = self.load_state_dict(torch.load(load_path, map_location=torch.device('cpu')))
        logger.info(f"missing_keys: {missing_keys}, unexpected_keys: {unexpected_keys}")


class MetaPathPredictor(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super(MetaPathPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.cls = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, num_labels)
        )
        # self.cls = nn.Linear(hidden_size * 2, num_labels)

    def forward(self, user_embeds, labels=None):
        """
        Args:
            user_embeds (batch_size, 2, hidden_size)
            labels (batch_size)
        """
        user_embeds = user_embeds.reshape(-1, self.hidden_size * 2)
        logits = self.cls(user_embeds)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return {
            'loss': loss,
            'logits': logits
        }

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'model.pt')
        torch.save(self.state_dict(), save_path)

    def load(self, load_dir):
        load_path = os.path.join(load_dir, 'model.pt')
        missing_keys, unexpected_keys = self.load_state_dict(torch.load(load_path, map_location=torch.device('cpu')))
        logger.info(f"missing_keys: {missing_keys}, unexpected_keys: {unexpected_keys}")


class BartForFlowGeneration(BartPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head.weight"]

    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = BartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        self.user_a_id = config.vocab_size - 2
        self.user_b_id = config.vocab_size - 1

    def get_input_embeds(self, input_ids, token_embeds, user_embeds):
        inputs_embeds = token_embeds[input_ids]  # (bs, seq_len, hs)

        user_a_mask = (input_ids == self.user_a_id)  # (bs, seq_len)
        user_a_idx = user_a_mask.nonzero()[:, 0]
        user_a_embed = user_embeds[:, 0, :]
        user_a_embeds = user_a_embed[user_a_idx]
        inputs_embeds[user_a_mask] = user_a_embeds.float()

        user_b_mask = (input_ids == self.user_b_id)  # (bs, seq_len)
        user_b_idx = user_b_mask.nonzero()[:, 0]
        user_b_embed = user_embeds[:, 1, :]
        user_b_embeds = user_b_embed[user_b_idx]
        inputs_embeds[user_b_mask] = user_b_embeds.float()

        return inputs_embeds

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None
    ) -> Dict[str, Any]:
        # 1. get encoder
        encoder = self.get_encoder()

        # 2. prepare encoder args and encoder kwargs from model kwargs
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache", "token", "user"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }

        # 3. make sure that encoder returns `ModelOutput`
        # model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs["return_dict"] = True
        # encoder_kwargs[model_input_name] = inputs_tensor
        encoder_kwargs['inputs_embeds'] = self.get_input_embeds(
            inputs_tensor, model_kwargs['token_embeds'], model_kwargs['user_embeds']
        )
        model_kwargs["encoder_outputs"]: ModelOutput = encoder(**encoder_kwargs)

        return model_kwargs

    @staticmethod
    def _expand_inputs_for_generation(
        input_ids: torch.LongTensor,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        attention_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[ModelOutput] = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        model_kwargs['user_embeds'] = model_kwargs['user_embeds'].index_select(0, expanded_return_idx)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(0, expanded_return_idx)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(0, expanded_return_idx)

        if is_encoder_decoder:
            if encoder_outputs is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
                0, expanded_return_idx.to(encoder_outputs.last_hidden_state.device)
            )
            model_kwargs["encoder_outputs"] = encoder_outputs
        return input_ids, model_kwargs

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        token_embeds=None,
        user_embeds=None,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        decoder_logits_mask=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "token_embeds": token_embeds,
            "user_embeds": user_embeds,
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "decoder_logits_mask": decoder_logits_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        token_embeds: torch.Tensor = None,
        user_embeds: torch.Tensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        decoder_logits_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        if encoder_outputs is None:
            inputs_embeds = self.get_input_embeds(input_ids=input_ids, token_embeds=token_embeds, user_embeds=user_embeds)
            input_ids = None

        decoder_inputs_embeds = self.get_input_embeds(
            input_ids=decoder_input_ids, token_embeds=token_embeds, user_embeds=user_embeds
        )
        decoder_input_ids = None

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
        if decoder_logits_mask is not None and labels is None:
            past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
            lm_logits[
                ~decoder_logits_mask[:, past_key_values_length: past_key_values_length + lm_logits.shape[1], :]] = -1e5

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.reshape(-1, self.config.vocab_size), labels.reshape(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past
