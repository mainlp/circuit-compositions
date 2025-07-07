import pickle

import torch

from comp_rep.models.nanoGPT import GPTConfig


def load_weights(
    self,
    w_q,
    w_k,
    w_v,
    w_o,
    up_proj,
    down_proj,
    b_q=None,
    b_k=None,
    b_v=None,
    b_o=None,
    b_up=None,
    b_down=None,
):
    self.attn.load_w_qkv(w_q, w_k, w_v, w_o, b_q, b_k, b_v, b_o)
    self.mlp.load_proj(up_proj, down_proj, b_up, b_down)


@torch.no_grad()
def load_everything(self, embeds_weights, pos_embeds_weights, block_weights):
    self.embedding.weight.data = embeds_weights
    self.position_embedding.weight.data = pos_embeds_weights
    for block, weights in zip(self.blocks, block_weights):
        block.load_weights(*weights)
    self.initialized = True


def format_weight_dict(weight_dict, n_layers):
    tok_embed = weight_dict["token_embed_embeddings"]
    pos_embed = weight_dict["pos_embed_embeddings"]
    blocks_embeds = []
    for layer_id in range(n_layers):
        k_w = weight_dict[f"transformer/layer_{layer_id}/attn/key_w"]
        k_b = weight_dict[f"transformer/layer_{layer_id}/attn/key_b"]
        q_w = weight_dict[f"transformer/layer_{layer_id}/attn/query_w"]
        q_b = weight_dict[f"transformer/layer_{layer_id}/attn/query_b"]
        v_w = weight_dict[f"transformer/layer_{layer_id}/attn/value_w"]
        v_b = weight_dict[f"transformer/layer_{layer_id}/attn/value_b"]
        o_w = weight_dict[f"transformer/layer_{layer_id}/attn/linear_w"]
        o_b = weight_dict[f"transformer/layer_{layer_id}/attn/linear_b"]
        up_w = weight_dict[f"transformer/layer_{layer_id}/mlp/linear_1_w"]
        up_b = weight_dict[f"transformer/layer_{layer_id}/mlp/linear_1_b"]
        down_w = weight_dict[f"transformer/layer_{layer_id}/mlp/linear_2_w"]
        down_b = weight_dict[f"transformer/layer_{layer_id}/mlp/linear_2_b"]

        blocks_embeds.append(
            [q_w, k_w, v_w, o_w, up_w, down_w, q_b, k_b, v_b, o_b, up_b, down_b]
        )
    return tok_embed, pos_embed, blocks_embeds


def get_key_value_vocab_and_model_size(tok_embed, blocks_embeds, num_heads):
    key_size = blocks_embeds[0][0].shape[1] // num_heads
    value_size = blocks_embeds[0][2].shape[1] // num_heads
    vocab_size = tok_embed.shape[0]
    model_size = tok_embed.shape[1]
    return key_size, value_size, vocab_size, model_size


def get_config_weights_and_vocab(input_path, pytorch=True, device=None, act="relu"):
    config_and_weights = pickle.load(open(input_path, "rb"))
    config_ = config_and_weights["config"]
    tok_embed, pos_embed, blocks_embeds = format_weight_dict(
        config_and_weights["model_params"], config_["num_layers"]
    )
    unembedding_mtx = (
        None
        if "unembedding_mtx" not in config_and_weights
        else config_and_weights["unembedding_mtx"]
    )
    key_size, value_size, vocab_size, model_size = get_key_value_vocab_and_model_size(
        tok_embed, blocks_embeds, config_["num_heads"]
    )
    print(f"Key size: {key_size}")
    print(f"Value size: {value_size}")
    print(f"Vocab size: {vocab_size}")
    print(f"Model size: {model_size}")
    print(f"Tok embed size: {tok_embed.shape}")
    print(f"Pos embed size: {pos_embed.shape}")
    print(f"unembed size: {unembedding_mtx.shape}")

    print(model_size)
    config = GPTConfig(
        block_size=model_size,
        n_embd=model_size,
        n_layer=config_["num_layers"],
        n_head=config_["num_heads"],
        key_size=key_size,
        value_size=value_size,
        mlp_hidden_size=config_["mlp_hidden_size"],
        vocab_size=vocab_size,
        act_func=act,
        layer_norm=config_["layer_norm"],
        causal=config_["causal"],
        max_position_embeddings=config_["max_seq_len"],
    )

    if pytorch:
        device = (
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        tok_embed = torch.tensor(tok_embed).to(device)
        pos_embed = torch.tensor(pos_embed).to(device)
        unembedding_mtx = (
            None
            if unembedding_mtx is None
            else torch.tensor(unembedding_mtx).to(device=device, dtype=tok_embed.dtype)
        )
        blocks_embeds = [
            [torch.tensor(w).to(device) for w in block] for block in blocks_embeds
        ]
    return (
        config,
        tok_embed,
        pos_embed,
        blocks_embeds,
        config_and_weights["vocab"],
        config_["bos"],
        config_["pad"],
        unembedding_mtx,
    )
