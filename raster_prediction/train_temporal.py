import os.path

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

from models import network
from data import temporal_dots

import yaml
from tqdm import tqdm


def main(config):

    writer = SummaryWriter()

    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model = getattr(network, config['model']['name'])(**config['model']['args']).to(device).train()
    print(f'Number of parameters: {torch.nn.utils.parameters_to_vector(model.parameters()).numel()}')

    dataset = getattr(temporal_dots, config['data']['name'])(**config['data']['args'])
    data_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.BCELoss(reduction='mean').to(device)

    losses = []

    for step, batch in tqdm(enumerate(data_loader)):

        batch = batch.to(device)
        seq_len = batch.shape[1]
        x_len = int(seq_len / 2)
        y_len = int(seq_len / 2)
        x = batch[:, :x_len, ...]
        y = batch[:, y_len:, ...]

        yp = model(x)
        loss = criterion(yp, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())

        if step % 25 == 0 and step != 0:
            avg_loss = sum(losses) / 25
            losses = []
            writer.add_scalar('train loss', avg_loss, step)
            writer.flush()

        if step % 1000 == 0 or step + 1 == len(data_loader):
            items = np.concatenate([dataset.render_item(torch.concat([x, yp], dim=1)[i]) for i in range(5)], axis=0)
            plt.imshow(items)
            path = "outputs/"
            path_exist = os.path.exists(path)
            if not path_exist:
                os.makedirs(path)
            plt.savefig(path + str(step).zfill(4) + ".png", bbox_inches='tight')


if __name__ == '__main__':

    with open("config_temporal.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    main(config)
