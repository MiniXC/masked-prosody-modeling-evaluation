from pathlib import Path
from collections import OrderedDict
import os

import yaml
import torch
from torch import nn
from transformers.utils.hub import cached_file
from rich.console import Console

console = Console()

from configs.args import BURNModelArgs
from scripts.util.remote import push_to_hub


class BreakProminenceClassifier(nn.Module):
    def __init__(
        self,
        args: BURNModelArgs,
    ):
        super().__init__()

        self.measures = args.measures.split(",")
        input_size = len(self.measures) * args.values_per_word

        if args.type == "mlp":
            self.mlp = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "layer_in_linear",
                            nn.Linear(input_size, args.hidden_dim),
                        ),
                        ("layer_in_gelu", nn.GELU()),
                        ("layer_in_dropout", nn.Dropout(args.dropout)),
                    ]
                )
            )

            for n in range(args.n_layers):
                self.mlp.add_module(
                    f"layer_{n}_linear", nn.Linear(args.hidden_dim, args.hidden_dim)
                )
                self.mlp.add_module(f"layer_{n}_gelu", nn.GELU())
                self.mlp.add_module(f"layer_{n}_dropout", nn.Dropout(args.dropout))

            self.mlp.add_module("layer_out_linear", nn.Linear(args.hidden_dim, 2))

        self.args = args

    def forward(self, x):
        if self.args.type == "mlp":
            return self.mlp(x)

    def save_model(self, path, accelerator=None, onnx=False):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if accelerator is not None:
            accelerator.save_model(self, path)
        else:
            torch.save(self.state_dict(), path / "pytorch_model.bin")
        if onnx:
            try:
                self.export_onnx(path / "model.onnx")
            except Exception as e:
                console.print(f"[red]Skipping ONNX export[/red]: {e}")
        with open(path / "model_config.yml", "w") as f:
            f.write(yaml.dump(self.args.__dict__, Dumper=yaml.Dumper))

    @staticmethod
    def from_pretrained(path_or_hubid):
        path = Path(path_or_hubid)
        if path.exists():
            config_file = path / "model_config.yml"
            model_file = path / "pytorch_model.bin"
        else:
            config_file = cached_file(path_or_hubid, "model_config.yml")
            model_file = cached_file(path_or_hubid, "pytorch_model.bin")
        args = yaml.load(open(config_file, "r"), Loader=yaml.Loader)
        args = BURNModelArgs(**args)
        model = BreakProminenceClassifier(args)
        model.load_state_dict(torch.load(model_file))
        return model

    @property
    def dummy_input(self):
        torch.manual_seed(0)
        return torch.randn(1, 256, self.args.values_per_word * len(self.measures))

    def export_onnx(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(
            self,
            self.dummy_input,
            path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
            opset_version=11,
        )
