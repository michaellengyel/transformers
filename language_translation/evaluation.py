import os
import torch
import importlib
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from train import get_ds, validation


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

    validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, cfg.seq_len, device, lambda msg: print(msg), num_examples=10)


if __name__ == '__main__':
    main()
