import os
import time
import importlib
from omegaconf import DictConfig, OmegaConf
import hydra

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import BilingualDataset

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from pathlib import Path


def validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, num_examples=5):

    model.eval()
    count = 0
    with torch.no_grad():

        for batch in validation_ds:
            count += 1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"
            model_out = model.greedy_decode(encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            print_msg('-'*80)
            print_msg(f'SOURCE: {source_text}')
            print_msg(f'TARGET: {target_text}')
            print_msg(f'PREDICTED: {model_out_text}')
            print_msg(f'RAW: {model_out.detach().cpu().numpy().tolist()}')

            if count == num_examples:
                break


def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]


def get_or_build_tokenizer(cfg, ds, lang):
    # cfg.tokenizer_file = ../tokenizer/tokenizer_{0}.json
    tokenizer_path = Path(cfg.tokenizer_file.format(lang))
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_ds(cfg):

    ds_raw = load_dataset('opus_books', f'{cfg.lang_src}-{cfg.lang_tgt}', split='train')

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(cfg, ds_raw, cfg.lang_src)
    tokenizer_tgt = get_or_build_tokenizer(cfg, ds_raw, cfg.lang_tgt)

    # Keep 90% for training and 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, cfg.lang_src, cfg.lang_tgt, cfg.seq_len)
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, cfg.lang_src, cfg.lang_tgt, cfg.seq_len)

    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][cfg.lang_src]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][cfg.lang_tgt]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    train_dataloader = DataLoader(train_ds, num_workers=cfg.num_workers, batch_size=cfg.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


@hydra.main(version_base=None, config_path="config", config_name="config.yaml")
def main(cfg : DictConfig):

    torch.manual_seed(1337)
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(cfg)
    writer = SummaryWriter(os.getcwd())

    module = importlib.import_module(cfg.module)
    model = getattr(module, cfg.model)(src_vocab_size=tokenizer_src.get_vocab_size(),
                                       tgt_vocab_size=tokenizer_tgt.get_vocab_size(),
                                       src_seq_len=cfg.seq_len,
                                       tgt_seq_len=cfg.seq_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-9)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    initial_epoch = 0
    global_step = 0

    for epoch in range(initial_epoch, cfg.num_epochs):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device)  # (B, seq_len)
            decoder_input = batch['decoder_input'].to(device)  # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device)  # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device)  # (B, 1, seq_len seq_len)
            label = batch['label'].to(device)  # (B, seq_len)

            # Run the tensors through the transformer
            encoder_output = model.encode(encoder_input, encoder_mask)  # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)  # (B, seq_len, d_model)
            proj_output = model.project(decoder_output)  # (B, seq_len, tgt_vocab_size)

            # (B, seq_len, tgt_vocab_size) -> (B * seq_len, tgt_vocab_size)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({f'loss': f'{loss.item():6.3f}'})

            # Log on tensorboard:
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Back-propagate the loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        # Run validation
        validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, cfg.seq_len, device, lambda msg: batch_iterator.write(msg))

        # Save model at the end of each epoch
        state = {'epoch': epoch,
                 'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'global_step': global_step}
        torch.save(state, "checkpoint.pth")
        print(f"Saved at: {Path('checkpoint.pth').resolve()}")
        time.sleep(1)


if __name__ == '__main__':
    main()
