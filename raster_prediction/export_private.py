import yaml
import torch
from models import network


def main(config):

    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')

    model = getattr(network, config['model']['name'])(**config['model']['args']).to(device).train()

    example_input = torch.zeros((1, 3, 64, 64)).to(device)
    torch.onnx.export(model, example_input, "onnx_filename.onnx", opset_version=11)


if __name__ == '__main__':
    with open("config.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    main(config)
