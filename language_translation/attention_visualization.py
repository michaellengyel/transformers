import torch
import hydra
import importlib
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import altair as alt
import pandas as pd
from train import get_ds


def load_next_batch(model, cfg, val_dataloader, tokenizer_src, tokenizer_tgt, device):
    # Load a sample batch from the validation set
    batch = next(iter(val_dataloader))
    encoder_input = batch["encoder_input"].to(device)
    encoder_mask = batch["encoder_mask"].to(device)
    decoder_input = batch["decoder_input"].to(device)
    decoder_mask = batch["decoder_mask"].to(device)

    encoder_input_tokens = [tokenizer_src.id_to_token(idx) for idx in encoder_input[0].cpu().numpy()]
    decoder_input_tokens = [tokenizer_tgt.id_to_token(idx) for idx in decoder_input[0].cpu().numpy()]

    # check that the batch size is 1
    assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

    model_out = model.greedy_decode(encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, cfg.seq_len, device)

    return batch, encoder_input_tokens, decoder_input_tokens, model_out


def mtx2df(m, max_row, max_col, row_tokens, col_tokens):
    return pd.DataFrame(
        [
            (
                r,
                c,
                float(m[r, c]),
                "%.3d %s" % (r, row_tokens[r] if len(row_tokens) > r else "<blank>"),
                "%.3d %s" % (c, col_tokens[c] if len(col_tokens) > c else "<blank>"),
            )
            for r in range(m.shape[0])
            for c in range(m.shape[1])
            if r < max_row and c < max_col
        ],
        columns=["row", "column", "value", "row_token", "col_token"],
    )


def get_attn_map(model, attn_type, layer, head):
    if attn_type == "encoder":
        attn = model.encoder.layers[layer].self_attention_block.attention_scores
    elif attn_type == "decoder":
        attn = model.decoder.layers[layer].self_attention_block.attention_scores
    elif attn_type == "encoder-decoder":
        attn = model.decoder.layers[layer].cross_attention_block.attention_scores
    return attn[0, head].data


def attn_map(model, attn_type, layer, head, row_tokens, col_tokens, max_sentence_len):
    df = mtx2df(
        get_attn_map(model, attn_type, layer, head),
        max_sentence_len,
        max_sentence_len,
        row_tokens,
        col_tokens,
    )
    return (
        alt.Chart(data=df)
        .mark_rect()
        .encode(
            x=alt.X("col_token", axis=alt.Axis(title="")),
            y=alt.Y("row_token", axis=alt.Axis(title="")),
            color="value",
            tooltip=["row", "column", "value", "row_token", "col_token"],
        )
        #.title(f"Layer {layer} Head {head}")
        .properties(height=400, width=400, title=f"Layer {layer} Head {head}")
        .interactive()
    )

def get_all_attention_maps(model, attn_type: str, layers: list[int], heads: list[int], row_tokens: list, col_tokens, max_sentence_len: int):
    charts = []
    for layer in layers:
        rowCharts = []
        for head in heads:
            rowCharts.append(attn_map(model, attn_type, layer, head, row_tokens, col_tokens, max_sentence_len))
        charts.append(alt.hconcat(*rowCharts))
    return alt.vconcat(*charts)


@hydra.main(version_base=None, config_path="config", config_name="config.yaml")
def main(cfg : DictConfig):

    torch.manual_seed(1337)
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(cfg)
    module = importlib.import_module(cfg.module)
    model = getattr(module, cfg.model)(src_vocab_size=tokenizer_src.get_vocab_size(),
                                       tgt_vocab_size=tokenizer_tgt.get_vocab_size(),
                                       src_seq_len=cfg.seq_len,
                                       tgt_seq_len=cfg.seq_len).to(device)

    state = torch.load(Path(hydra.utils.get_original_cwd(), cfg.weights))
    model.load_state_dict(state['model_state_dict'])

    batch, encoder_input_tokens, decoder_input_tokens, model_out = load_next_batch(model, cfg, val_dataloader, tokenizer_src, tokenizer_tgt, device)
    print(f'Source: {batch["src_text"][0]}')
    print(f'Target: {batch["tgt_text"][0]}')
    print(f'Predicted: {tokenizer_tgt.decode(model_out.detach().cpu().numpy())}')
    sentence_len = encoder_input_tokens.index("[PAD]")

    layers = [0, 1, 2]
    heads = [0, 1, 2, 3, 4, 5, 6, 7]

    # alt.renderers.enable('altair_viewer')

    # Encoder Self-Attention
    chart_encoder = get_all_attention_maps(model, "encoder", layers, heads, encoder_input_tokens, encoder_input_tokens, min(20, sentence_len))
    # Encoder Self-Attention
    chart_decoder = get_all_attention_maps(model, "decoder", layers, heads, decoder_input_tokens, decoder_input_tokens, min(20, sentence_len))
    # Encoder Self-Attention
    chart_encoder_decoder = get_all_attention_maps(model, "encoder-decoder", layers, heads, encoder_input_tokens, decoder_input_tokens, min(20, sentence_len))

    chart_encoder.save('encoder.html')
    chart_decoder.save('decoder.html')
    chart_encoder_decoder.save('encoder_decoder.html')


if __name__ == '__main__':
    main()
