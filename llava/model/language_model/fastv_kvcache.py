import torch
from typing import Tuple, Callable
from transformers import AutoConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaModel, Cache, DynamicCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.models.llama import LlamaConfig
from transformers import Qwen2Config, Qwen2Model
from transformers.modeling_outputs import BaseModelOutputWithPast
from typing import List, Optional, Tuple, Union

USE_SEPARATE_R_FOR_GLOBAL_AND_LOCAL = False
K = 2
ratio_global = 0.75
ratio_local = 0.75
num_global_image_tokens = 729

class FastVModelMixin:
    """
    A Mixin (or base class) containing the shared forward logic for 
    both FastVLlamaModel and FastVQwen2Model.
    """

    def _forward_shared(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor],
        past_key_values: Optional[List[torch.FloatTensor]],
        inputs_embeds: Optional[torch.FloatTensor],
        use_cache: Optional[bool],
        output_attentions: Optional[bool],
        output_hidden_states: Optional[bool],
        return_dict: Optional[bool],
        cache_position: Optional[torch.LongTensor],
        # custom arguments
        num_image_tokens_per_image: Optional[int],
        image_token_indices_for_each_batch: Optional[torch.Tensor],
    ):
        """
        Contains the shared logic that was duplicated across FastVLlamaModel and FastVQwen2Model.
        Simply parameterize the small differences like K and ratio.
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # handle gradient checkpointing ...
        if self.gradient_checkpointing and self.training and use_cache:
            print("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
            use_cache = False

        # prepare caches
        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        # set position_ids
        if position_ids is None:
            device = (
                input_ids.device if input_ids is not None else inputs_embeds.device
            )
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device
            ).unsqueeze(0)

        # embed tokens
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # prepare attention mask
        # (this is identical in both classes, so we factor it out here)
        if False:
            # if self._supports_flash_attn_2:
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self.config._attn_implementation == "sdpa" and not output_attentions:
            # 2D vs 4D mask logic
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )
        else:
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        # run through layers
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                # image pruning logic
                if (
                    decoder_layer.self_attn.layer_idx == K
                    and seq_length > 1
                    and image_token_indices_for_each_batch is not None
                ):
                    device = hidden_states.device

                    # Start indices, ignoring the first and last
                    image_start_indices = image_token_indices_for_each_batch[0][1:-1]

                    keep_indices = []
                    for image_start_index in image_start_indices:


                        if USE_SEPARATE_R_FOR_GLOBAL_AND_LOCAL:

                            # define global and local image token indices
                            num_local_image_tokens = num_image_tokens_per_image - num_global_image_tokens
                            global_start_index = image_start_index
                            global_end_index = image_start_index + num_global_image_tokens
                            local_start_index = global_end_index
                            local_end_index = image_start_index + num_image_tokens_per_image

                            # compute mean attention of global image tokens
                            image_attention_score_global = self.last_attention.mean(dim=1)[0][-1][
                                global_start_index : global_end_index
                            ]

                            # compute mean attention of local image tokens
                            image_attention_score_local = self.last_attention.mean(dim=1)[0][-1][
                                local_start_index : local_end_index
                            ]

                            # pick top ratio of global image tokens
                            top_attention_rank_index_global = (
                                image_attention_score_global.topk(int(num_global_image_tokens * ratio_global)).indices
                                + image_start_index
                            )

                            # pick top ratio of local image tokens
                            top_attention_rank_index_local = (
                                image_attention_score_local.topk(int(num_local_image_tokens * ratio_local)).indices
                                + image_start_index
                            )

                            # append indices of top global and local image tokens
                            keep_indices.append(top_attention_rank_index_global)
                            keep_indices.append(top_attention_rank_index_local)
                        else:
                            # compute mean attention
                            image_attention_score = self.last_attention.mean(dim=1)[0][-1][
                                image_start_index : image_start_index + num_image_tokens_per_image
                            ]
                            # pick top ratio of them
                            top_attention_rank_index = (
                                image_attention_score.topk(int(num_image_tokens_per_image * ratio_global)).indices
                                + image_start_index
                            )
                            keep_indices.append(top_attention_rank_index)

                    # add non-image tokens
                    keep_indices.append(torch.arange(0, image_start_indices[0], device=device))
                    keep_indices.append(torch.arange(
                        image_start_indices[-1] + num_image_tokens_per_image,
                        seq_length,
                        device=device
                    ))

                    keep_indices = torch.cat(keep_indices).sort().values

                    # filter hidden states
                    hidden_states = hidden_states[:, keep_indices, :]

                    # adjust attention mask
                    if attention_mask is not None:
                        # ... for simplicity, you may need to re-index it carefully
                        attention_mask = attention_mask[:, :, :hidden_states.shape[1], :hidden_states.shape[1]]

                    # update position_ids
                    position_ids = torch.arange(
                        0,
                        hidden_states.shape[1],
                        device=device
                    ).unsqueeze(0)

                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                    )

                elif decoder_layer.self_attn.layer_idx == K - 1:
                    # capture the attention weights for next layer
                    temp_layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=True,
                        use_cache=use_cache,
                    )
                    self.last_attention = temp_layer_outputs[1]
                    layer_outputs = temp_layer_outputs
                else:
                    # if attention_mask is None:
                    #     attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                    #         attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
                    #     )
                    if decoder_layer.self_attn.layer_idx == K and position_ids.shape[1] == 1:
                        position_ids[0][0] = past_key_values.get_usable_length(hidden_states.shape[-2], decoder_layer.self_attn.layer_idx)
                        # attention_mask = attention_mask[:, :, :position_ids.item() + 1, :position_ids.item() + 1]
                    # normal
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                    )

            hidden_states = layer_outputs[0]

            # handle caching
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add final hidden states
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
            )

        # return either tuple or dataclass
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

class FastVLlamaModel(LlamaModel, FastVModelMixin):
    def __init__(self, config: LlamaConfig):
        self.last_attention = None
        super().__init__(config)
        # any other init logic

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_image_tokens_per_image: Optional[int] = None,
        image_token_indices_for_each_batch: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        return self._forward_shared(
            K=K,
            ratio_global=ratio_global,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            num_image_tokens_per_image=num_image_tokens_per_image,
            image_token_indices_for_each_batch=image_token_indices_for_each_batch,
        )

class FastVQwen2Model(Qwen2Model, FastVModelMixin):
    def __init__(self, config: Qwen2Config):
        config._attn_implementation = "sdpa"
        config._attn_implementation_internal = "sdpa"
        self.last_attention = None
        super().__init__(config)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_image_tokens_per_image: Optional[int] = None,
        image_token_indices_for_each_batch: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        # Here you only call the shared logic, specifying K=3, ratio=0.5
        return self._forward_shared(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            num_image_tokens_per_image=num_image_tokens_per_image,
            image_token_indices_for_each_batch=image_token_indices_for_each_batch,
        )


def _prepare_4d_causal_attention_mask(
    attention_mask: Optional[torch.Tensor],
    input_shape: Union[torch.Size, Tuple, List],
    inputs_embeds: torch.Tensor,
    past_key_values_length: int,
    sliding_window: Optional[int] = None,
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`

    Args:
        attention_mask (`torch.Tensor` or `None`):
            A 2D attention mask of shape `(batch_size, key_value_length)`
        input_shape (`tuple(int)` or `list(int)` or `torch.Size`):
            The input shape should be a tuple that defines `(batch_size, query_length)`.
        inputs_embeds (`torch.Tensor`):
            The embedded inputs as a torch Tensor.
        past_key_values_length (`int`):
            The length of the key value cache.
        sliding_window (`int`, *optional*):
            If the model uses windowed attention, a sliding window should be passed.
    """
    attn_mask_converter = AttentionMaskConverter(is_causal=True, sliding_window=sliding_window)

    key_value_length = input_shape[-1] + past_key_values_length

    # 4d mask is passed through the layers
    if attention_mask is not None and len(attention_mask.shape) == 2:
        attention_mask = attn_mask_converter.to_4d(
            attention_mask, input_shape[-1], key_value_length=key_value_length, dtype=inputs_embeds.dtype
        )
    elif attention_mask is not None and len(attention_mask.shape) == 4:
        expected_shape = (input_shape[0], 1, input_shape[1], key_value_length)
        if tuple(attention_mask.shape) != expected_shape:
            raise ValueError(
                f"Incorrect 4D attention_mask shape: {tuple(attention_mask.shape)}; expected: {expected_shape}."
            )
        else:
            # if the 4D mask has correct shape - invert it and fill with negative infinity
            inverted_mask = 1.0 - attention_mask
            attention_mask = inverted_mask.masked_fill(
                inverted_mask.to(torch.bool), torch.finfo(inputs_embeds.dtype).min
            )
    else:
        attention_mask = attn_mask_converter.to_causal_4d(
            input_shape[0], input_shape[-1], key_value_length, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )

    return attention_mask


# Adapted from _prepare_4d_causal_attention_mask
def _prepare_4d_causal_attention_mask_for_sdpa(
    attention_mask: Optional[torch.Tensor],
    input_shape: Union[torch.Size, Tuple, List],
    inputs_embeds: torch.Tensor,
    past_key_values_length: int,
    sliding_window: Optional[int] = None,
):
    """
    Prepares the correct `attn_mask` argument to be used by `torch.nn.functional.scaled_dot_product_attention`.

    In case no token is masked in the `attention_mask` argument, we simply set it to `None` for the cases `query_length == 1` and
    `key_value_length == query_length`, and rely instead on SDPA `is_causal` argument to use causal/non-causal masks,
    allowing to dispatch to the flash attention kernel (that can otherwise not be used if a custom `attn_mask` is passed).
    """
    attn_mask_converter = AttentionMaskConverter(is_causal=True, sliding_window=sliding_window)

    key_value_length = input_shape[-1] + past_key_values_length
    batch_size, query_length = input_shape

    # torch.jit.trace, symbolic_trace and torchdynamo with fullgraph=True are unable to capture the controlflow `is_causal=attention_mask is None and q_len > 1`
    # used as an SDPA argument. We keep compatibility with these tracing tools by always using SDPA's `attn_mask` argument in case we are tracing.
    # TODO: For dynamo, rather use a check on fullgraph=True once this is possible (https://github.com/pytorch/pytorch/pull/120400).
    is_tracing = (
        torch.jit.is_tracing()
        or isinstance(inputs_embeds, torch.fx.Proxy)
        or (hasattr(torch, "_dynamo") and torch._dynamo.is_compiling())
    )

    if attention_mask is not None:
        # 4d mask is passed through
        if len(attention_mask.shape) == 4:
            expected_shape = (input_shape[0], 1, input_shape[1], key_value_length)
            if tuple(attention_mask.shape) != expected_shape:
                raise ValueError(
                    f"Incorrect 4D attention_mask shape: {tuple(attention_mask.shape)}; expected: {expected_shape}."
                )
            else:
                # if the 4D mask has correct shape - invert it and fill with negative infinity
                inverted_mask = 1.0 - attention_mask.to(inputs_embeds.dtype)
                attention_mask = inverted_mask.masked_fill(
                    inverted_mask.to(torch.bool), torch.finfo(inputs_embeds.dtype).min
                )
                return attention_mask

        elif not is_tracing and torch.all(attention_mask == 1):
            if query_length == 1:
                # For query_length == 1, causal attention and bi-directional attention are the same.
                attention_mask = None
            elif key_value_length == query_length:
                attention_mask = None
            else:
                # Unfortunately, for query_length > 1 and key_value_length != query_length, we cannot generally ignore the attention mask, as SDPA causal mask generation
                # may be wrong. We will set `is_causal=False` in SDPA and rely on Transformers attention_mask instead, hence not setting it to None here.
                # Reference: https://github.com/pytorch/pytorch/issues/108108
                pass
    elif query_length > 1 and key_value_length != query_length:
        # See the comment above (https://github.com/pytorch/pytorch/issues/108108).
        # Ugly: we set it to True here to dispatch in the following controlflow to `to_causal_4d`.
        attention_mask = True
    elif is_tracing:
        raise ValueError(
            'Attention using SDPA can not be traced with torch.jit.trace when no attention_mask is provided. To solve this issue, please either load your model with the argument `attn_implementation="eager"` or pass an attention_mask input when tracing the model.'
        )

    if attention_mask is None:
        expanded_4d_mask = None
    elif attention_mask is True:
        expanded_4d_mask = attn_mask_converter.to_causal_4d(
            input_shape[0], input_shape[-1], key_value_length, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
    else:
        expanded_4d_mask = attn_mask_converter.to_4d(
            attention_mask,
            input_shape[-1],
            dtype=inputs_embeds.dtype,
            key_value_length=key_value_length,
        )

        # Attend to all tokens in masked rows from the causal_mask, for example the relevant first rows when
        # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
        # Details: https://github.com/pytorch/pytorch/issues/110213
        if not is_tracing and expanded_4d_mask.device.type == "cuda":
            expanded_4d_mask = AttentionMaskConverter._unmask_unattended(
                expanded_4d_mask, min_dtype=torch.finfo(inputs_embeds.dtype).min
            )

    return expanded_4d_mask