from pathlib import Path
from collections import OrderedDict
import os

import yaml
import torch
from torch import nn
from transformers.utils.hub import cached_file
from rich.console import Console

console = Console()

from configs.args import TIMITModelArgs
from scripts.util.remote import push_to_hub
from model.modules import ConformerLayer, PositionalEncoding, TransformerEncoder


class PhonemeWordBoundaryClassifier(nn.Module):
    def __init__(
        self,
        args: TIMITModelArgs,
    ):
        super().__init__()

        if not args.use_mpm:
            self.measures = args.measures.split(",")
            if not args.use_cwt:
                input_size = len(self.measures) * 2
            else:
                input_size = args.cwt_n_bins * 2 * len(self.measures)
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
        elif args.type == "conformer":
            self.input_linear = nn.Linear(input_size, args.hidden_dim)

            self.positional_encoding = PositionalEncoding(args.hidden_dim)

            self.conformer = TransformerEncoder(
                ConformerLayer(
                    args.hidden_dim,
                    args.n_heads,
                    conv_in=args.hidden_dim,
                    conv_filter_size=args.filter_size,
                    conv_kernel=(args.kernel_size, 1),
                    batch_first=True,
                    dropout=args.dropout,
                ),
                num_layers=args.n_layers,
            )

            self.output_layer = nn.Sequential(
                nn.Linear(args.hidden_dim, args.hidden_dim),
                nn.LayerNorm((args.max_length, args.hidden_dim)),
                nn.GELU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.hidden_dim, 2),
            )

        self.args = args

    def forward(self, x):
        if self.args.type == "mlp":
            return self.mlp(x)
        elif self.args.type == "linear":
            return self.linear(x)
        elif self.args.type == "conformer":
            x = self.input_linear(x)
            x = self.positional_encoding(x)
            x = self.conformer(x)
            x = self.output_layer(x)
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
        args = TIMITModelArgs(**args)
        model = PhonemeWordBoundaryClassifier(args)
        model.load_state_dict(torch.load(model_file))
        return model

    @property
    def dummy_input(self):
        torch.manual_seed(0)
        if not self.args.use_mpm:
            if not self.args.use_cwt:
                return torch.randn(1, 384, len(self.measures) * 2)
            else:
                return torch.randn(1, 384, self.args.cwt_n_bins * 2 * len(self.measures))
        else:
            return torch.randn(1, 384, 512)

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
