from pathlib import Path
from dataclasses import dataclass

import yaml
import torch
from torch import nn
from transformers.utils.hub import cached_file
from rich.console import Console

console = Console()

from .modules import (
    PositionalEncoding,
    TransformerEncoder,
    ConformerLayer,
)


@dataclass
class ModelArgs:
    n_layers: int = 8
    n_heads: int = 2
    kernel_size: int = 7
    filter_size: int = 256
    hidden_dim: int = 512
    dropout: float = 0.1


class MaskedProsodyModel(nn.Module):
    def __init__(
        self,
        args,
    ):
        super().__init__()

        bins = args.bins

        self.pitch_embedding = nn.Embedding(bins + 2, args.filter_size)
        self.energy_embedding = nn.Embedding(bins + 2, args.filter_size)
        self.vad_embedding = nn.Embedding(bins + 2, args.filter_size)

        self.positional_encoding = PositionalEncoding(args.filter_size)

        self.transformer = TransformerEncoder(
            ConformerLayer(
                args.filter_size,
                args.n_heads,
                conv_in=args.filter_size,
                conv_filter_size=args.filter_size,
                conv_kernel=(args.kernel_size, 1),
                batch_first=True,
                dropout=args.dropout,
            ),
            num_layers=args.n_layers,
        )

        self.output_layer = nn.Sequential(
            nn.Linear(args.filter_size, args.filter_size),
            nn.LayerNorm((args.max_length, args.filter_size)),
            nn.GELU(),
        )

        self.output_pitch = nn.Sequential(
            nn.Linear(args.filter_size, args.filter_size),
            nn.LayerNorm((args.max_length, args.filter_size)),
            nn.GELU(),
            nn.Linear(args.filter_size, bins),
        )

        self.output_energy = nn.Sequential(
            nn.Linear(args.filter_size, args.filter_size),
            nn.LayerNorm((args.max_length, args.filter_size)),
            nn.GELU(),
            nn.Linear(args.filter_size, bins),
        )

        self.output_vad = nn.Sequential(
            nn.Linear(args.filter_size, args.filter_size),
            nn.LayerNorm((args.max_length, args.filter_size)),
            nn.GELU(),
            nn.Linear(args.filter_size, bins),
        )

        self.apply(self._init_weights)

        self.args = args

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, return_layer=None):
        pitch = x[:, 0]
        energy = x[:, 1]
        vad = x[:, 2]
        pitch = self.pitch_embedding(pitch)
        energy = self.energy_embedding(energy)
        vad = self.vad_embedding(vad)
        x = pitch + energy + vad
        x = self.positional_encoding(x)
        if return_layer is not None:
            x, reprs = self.transformer(x, return_layer=return_layer)
        else:
            x = self.transformer(x)
        x = self.output_layer(x)
        pitch = self.output_pitch(x)
        energy = self.output_energy(x)
        vad = self.output_vad(x)
        if return_layer is not None:
            return {
                "y": torch.stack(
                    [
                        pitch,
                        energy,
                        vad,
                    ]
                ).transpose(0, 1),
                "representations": reprs,
            }
        else:
            return torch.stack(
                [
                    pitch,
                    energy,
                    vad,
                ]
            ).transpose(0, 1)

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
        model_specific_args = [
            "bins",
            "max_length",
        ]
        fargs = {k: v for k, v in args.items() if k not in model_specific_args}
        margs = ModelArgs(**fargs)
        margs.bins = args["bins"]
        margs.max_length = args["max_length"]
        model = MaskedProsodyModel(margs)
        model.load_state_dict(torch.load(model_file))
        return model

    @property
    def dummy_input(self):
        torch.manual_seed(0)
        return torch.stack(
            [
                torch.randint(0, self.args.bins, (1, self.args.max_length)),
                torch.randint(0, self.args.bins, (1, self.args.max_length)),
                torch.randint(0, self.args.bins, (1, self.args.max_length)),
            ]
        ).transpose(0, 1)

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
