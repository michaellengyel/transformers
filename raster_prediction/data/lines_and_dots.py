import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np

from PIL import Image, ImageDraw


class RgbDots(Dataset):
    def __init__(self, num_samples, data_shape):
        self.num_samples = num_samples
        self.data_shape = data_shape
        self.transform = transforms.ToTensor()
        self.parameters = self.generate_parameters(num_samples, data_shape)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):

        x_canvas = Image.new('RGB', self.data_shape, color="gray")
        y_canvas = Image.new('RGB', self.data_shape, color="gray")
        self.draw_circles(idx, x_canvas, y_canvas)
        x = self.transform(x_canvas)
        y = self.transform(y_canvas)
        return x, y

    def generate_parameters(self, num_samples, data_shape):
        np.random.seed(1337)
        random_integers = np.random.randint(0, data_shape[0], size=(num_samples, 6))
        random_integers[:, 2:4] = (random_integers[:, 0:2] + random_integers[:, 4:6]) / 2
        return random_integers.tolist()

    def draw_circles(self, idx, x, y):

        draw_x = ImageDraw.Draw(x)
        draw_y = ImageDraw.Draw(y)

        circles = self.parameters[idx]
        circles = [circles[i:i + 2] for i in range(0, len(circles), 2)]

        r = 2
        circle = circles[0]
        draw_x.ellipse((circle[0] - r, circle[1] - r, circle[0] + r, circle[1] + r), fill="black")
        circle = circles[2]
        draw_x.ellipse((circle[0] - r, circle[1] - r, circle[0] + r, circle[1] + r), fill="black")

        # circle = circles[2]
        # draw_y.ellipse((circle[0] - r, circle[1] - r, circle[0] + r, circle[1] + r), fill="blue")
        shape = [(circles[0][0], circles[0][1]), (circles[2][0], circles[2][1])]
        draw_y.line(shape, fill="black", width=2)

    def render_item(self, x, y, yp):
        item = torch.concat([x, y, yp], dim=-1).permute(1, 2, 0).detach().cpu().numpy()
        return item


from matplotlib import pyplot as plt
if __name__ == '__main__':

    dataset = RgbDots(1000, (64, 64))
    data_loader = DataLoader(dataset, batch_size=30, shuffle=True)

    for batch in data_loader:
        items = np.concatenate([dataset.render_item(batch[0][i], batch[1][i], batch[1][i]) for i in range(3)], axis=0)
        plt.imshow(items)
        plt.tight_layout()
        plt.show()
