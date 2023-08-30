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

        if not args.use_mpm:
            self.measures = args.measures.split(",")
            input_size = len(self.measures) * args.values_per_word
        else:
            input_size = 512

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
        elif args.type == "linear":
            self.linear = nn.Linear(input_size, 2)
        elif args.type == "lstm":
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=args.hidden_dim,
                num_layers=args.n_layers,
                bidirectional=True,
                batch_first=True,
            )
            self.linear = nn.Linear(args.hidden_dim * 2, 2)

        self._init_weights(self)

        self.args = args

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)

    def forward(self, x):
        if self.args.type == "mlp":
            return self.mlp(x)
        elif self.args.type == "linear":
            return self.linear(x)
        elif self.args.type == "lstm":
            x, _ = self.lstm(x)
            x = self.linear(x)
            return x

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
        if not self.args.use_mpm:
            return torch.randn(1, 256, self.args.values_per_word * len(self.measures))
        else:
            return torch.randn(1, 256, 512)

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
