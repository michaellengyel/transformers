import os
import torch
import importlib
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from torch.utils.data import DataLoader
from dataset import ShakespearDataset

@hydra.main(version_base=None, config_path="config", config_name="config.yaml")
def main(cfg : DictConfig):

    torch.manual_seed(1337)
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')

    val_dataset = ShakespearDataset(**cfg.train.dataset)
    val_dataloader = DataLoader(val_dataset, **cfg.train.loader)

    module = importlib.import_module(cfg.module)
    model = getattr(module, cfg.model)(vocab_size=val_dataset.get_vocab_size(), **cfg.model_args).to(device)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())}")
    model.eval()

    example_input = next(iter(val_dataloader))[0].to(device)
    onnx_file_path = "simple_model.onnx"

    torch.onnx.export(
        model,  # Model to export
        example_input,  # Example input
        onnx_file_path,  # File path to save the ONNX file
        export_params=True,  # Store the trained parameter weights
        opset_version=13,  # ONNX opset version
        do_constant_folding=True,  # Optimize constant folding for ONNX
        input_names=["input"],  # Input tensor name
        output_names=["output"],  # Output tensor name
        dynamic_axes={  # Allow variable batch size
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )


if __name__ == '__main__':
    main()
