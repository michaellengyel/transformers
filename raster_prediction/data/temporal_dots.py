import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import math

from PIL import Image, ImageDraw


class TemporalDots(Dataset):
    def __init__(self, num_samples, data_shape, past_seq_len, pred_seq_len):
        self.num_samples = num_samples
        self.data_shape = data_shape
        self.past_seq_len = past_seq_len
        self.pred_seq_len = pred_seq_len
        self.full_seq_len = past_seq_len + pred_seq_len

        self.transform = transforms.ToTensor()
        self.parameters = self.generate_parameters(num_samples, data_shape)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):

        canvas = [Image.new('L', self.data_shape, color="black") for _ in range(self.full_seq_len)]
        self.draw_circles(idx, canvas)
        canvas = [self.transform(x).unsqueeze(0) for x in canvas]
        canvas = torch.concat(canvas, dim=0)
        return canvas

    def generate_parameters(self, num_samples, data_shape):
        rng = np.random.default_rng(1337)
        start = rng.random(size=(num_samples, 2)) * data_shape[0]
        end = rng.random(size=(num_samples, 2)) * data_shape[0]
        delta = (end - start) / self.full_seq_len

        points = [start]
        for i in range(1, self.full_seq_len, 1):
            start = start+delta
            points.append(start)

        points = np.concatenate(points, axis=1)
        return points.tolist()

    def draw_circles(self, idx, canvas):
        r = 3.0  # 1.5
        circles = self.parameters[idx]
        for i, raster in enumerate(canvas):

            draw = ImageDraw.Draw(raster)
            circle = circles[i*2:i*2 + 2]
            draw.ellipse((circle[0] - r, circle[1] - r, circle[0] + r, circle[1] + r), fill="white")

    def render_item(self, x):
        s, c, w, h = x.shape

        # 1.
        # item = x.squeeze().reshape(s*w, w).permute(1, 0).detach().cpu().numpy()

        # 2.
        # item = [x[i] * math.log(i+2) for i in range(s)]
        # item = torch.concat(item, dim=0).sum(dim=0)

        # 3.
        item = [x[i].detach().cpu() for i in range(s)]
        item = torch.concat(item, dim=0).sum(dim=0)

        # Normalize
        outmap_min = item.min()
        outmap_max = item.max()
        item = (item - outmap_min) / (outmap_max - outmap_min)

        return item


from matplotlib import pyplot as plt
if __name__ == '__main__':

    dataset = TemporalDots(num_samples=1000, data_shape=(64, 64), past_seq_len=6, pred_seq_len=6)
    data_loader = DataLoader(dataset, batch_size=30, shuffle=True)

    for batch in data_loader:
        items = np.concatenate([dataset.render_item(batch[i]) for i in range(5)], axis=0)
        plt.imshow(items)
        plt.tight_layout()
        plt.show()
        print("Batch shape:", batch.shape)  # Shape: (batch_size, channels, height, width)
