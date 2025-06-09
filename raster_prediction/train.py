import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

from models import network, conv_vit_model
from data import rgb_dots, lines_and_dots

import yaml
from tqdm import tqdm


def main(config):

    writer = SummaryWriter()

    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model = getattr(network, config['model']['name'])(**config['model']['args']).to(device).train()
    # model = conv_vit_model.TransformerEncoder(src_seq_len=169, d_model=64, N=8, h=8, dropout=0.1, d_ff=128).to(device).train()

    print(f'Number of parameters: {torch.nn.utils.parameters_to_vector(model.parameters()).numel()}')

    dataset = getattr(rgb_dots, config['data']['name'])(**config['data']['args'])
    data_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.MSELoss(reduction='mean').to(device)

    losses = []

    for step, batch in tqdm(enumerate(data_loader)):

        x = batch[0].to(device)
        y = batch[1].to(device)

        yp = model(x)
        loss = criterion(yp, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if step % 25 == 0 and step != 0:
            avg_loss = sum(losses) / 25
            losses = []
            writer.add_scalar('train loss', avg_loss, step)
            writer.flush()

        if step % 500 == 0 or step + 1 == len(data_loader):
            items = np.concatenate([dataset.render_item(x[i], y[i], yp[i]) for i in range(3)], axis=0)
            plt.imshow(items)
            plt.savefig("./out/" + str(step).zfill(4) + ".png", bbox_inches='tight')


if __name__ == '__main__':

    with open("config.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    main(config)
