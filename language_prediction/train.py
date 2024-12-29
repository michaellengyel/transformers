import os
import importlib
import numpy as np
import torch
from tqdm import tqdm
import itertools
import logging
import torch.nn as nn
from torch.nn import functional as F

from omegaconf import DictConfig, OmegaConf
import hydra

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import ShakespearDataset

@torch.no_grad()
def estimate_loss(model, val_loader, device):
    losses = []
    model.eval()
    num_eval = 500
    for i, (x, y) in enumerate(val_loader):
        x = x.to(device)
        y = y.to(device)
        logits, loss = model(x, y)
        losses.append(loss.item())
        if i >= num_eval:
            break
    model.train()
    return np.array(losses).mean()

@hydra.main(version_base=None, config_path="config", config_name="config.yaml")
def main(cfg : DictConfig):

    torch.manual_seed(1337)

    writer = SummaryWriter(os.getcwd())
    logger = logging.getLogger(__name__)
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')

    train_dataset = ShakespearDataset(**cfg.train.dataset)
    val_dataset = ShakespearDataset(**cfg.val.dataset)
    train_dataloader = DataLoader(train_dataset, **cfg.train.loader)
    val_dataloader = DataLoader(val_dataset, **cfg.val.loader)

    module = importlib.import_module(cfg.module)
    model = getattr(module, cfg.model)(vocab_size=train_dataset.get_vocab_size(), **cfg.model_args).to(device)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    endless_dataloader = itertools.cycle(train_dataloader)

    for i in tqdm(range(cfg.max_iters)):
        x, y = next(endless_dataloader)
        x = x.to(device)
        y = y.to(device)
        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        writer.add_scalar('train/loss', loss.item(), i)

        if i % cfg.eval_interval == 0 and i != 0:
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
            logger.info(train_dataset.decode(model.generate(context, max_new_tokens=500)[0].tolist()))
            val_dataloader = DataLoader(val_dataset, **cfg.val.loader)
            val_loss = estimate_loss(model, val_dataloader, cfg.device)
            writer.add_scalar('val/loss', val_loss.item(), i)


if __name__ == '__main__':
    main()
