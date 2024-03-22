import os
import sys
from collections import deque
from pathlib import Path

sys.path.append(".")  # add root of project to path

# torch & hf
import torch
from torch.utils.data import DataLoader
import torchvision
from accelerate import Accelerator
from transformers import get_linear_schedule_with_warmup, HfArgumentParser
from datasets import load_dataset

# logging & etc
from torchinfo import summary
from torchview import draw_graph
import wandb
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm.auto import tqdm
import yaml
from rich.console import Console
import matplotlib.pyplot as plt

console = Console()

# local imports
from configs.args import TrainingArgs, BURNModelArgs, BURNCollatorArgs as CollatorArgs
from configs.validation import validate_args
from util.remote import wandb_update_config, wandb_init, push_to_hub
from util.plotting import plot_baseline_burn_batch as plot_baseline_batch
from util.plotting import plot_prosody_model_burn_batch as plot_prosody_model_batch
from model.burn_classifiers import BreakProminenceClassifier
from collators import get_collator


no_results = {
            "loss": 100000,
            "break_loss": 100000,
            "prom_loss": 100000,
            "prom_acc": 0,
            "prom_f1": 0,
            "prom_f1_binary": 0,
            "prom_precision": 0,
            "prom_recall": 0,
            "break_acc": 0,
            "break_f1": 0,
            "break_f1_binary": 0,
            "break_precision": 0,
            "break_recall": 0,
            "best_prom_threshold": 0,
            "best_break_threshold": 0,
            }

def print_and_draw_model():
    dummy_input = model.dummy_input
    # repeat dummy input to match batch size (regardless of how many dimensions)
    dummy_input = dummy_input.repeat(
        (training_args.batch_size,) + (1,) * (len(dummy_input.shape) - 1)
    )
    console_print(f"[green]input shape[/green]: {dummy_input.shape}")
    model_summary = summary(
        model,
        input_data=dummy_input,
        verbose=0,
        col_names=[
            "input_size",
            "output_size",
            "num_params",
        ],
    )
    console_print(model_summary)
    if accelerator.is_main_process:
        model_graph = draw_graph(
            model,
            input_data=dummy_input,
            save_graph=True,
            directory="figures/",
            filename="model_burn",
        )


def console_print(*args, **kwargs):
    if accelerator.is_main_process:
        console.print(*args, **kwargs)


def console_rule(*args, **kwargs):
    if accelerator.is_main_process:
        console.rule(*args, **kwargs)


def wandb_log(prefix, log_dict, round_n=3, print_log=True):
    if accelerator.is_main_process:
        log_dict = {f"{prefix}/{k}": v for k, v in log_dict.items()}
        wandb.log(log_dict, step=global_step)
        if print_log:
            log_dict = {k: round(v, round_n) for k, v in log_dict.items()}
            console.log(log_dict)


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def save_checkpoint():
    accelerator.wait_for_everyone()
    checkpoint_name = training_args.run_name
    checkpoint_path = (
        Path(training_args.checkpoint_path) / checkpoint_name / f"step_{global_step}"
    )
    # model
    model.save_model(checkpoint_path, accelerator, onnx=training_args.save_onnx)
    if accelerator.is_main_process:
        # training args
        with open(checkpoint_path / "training_args.yml", "w") as f:
            f.write(yaml.dump(training_args.__dict__, Dumper=yaml.Dumper))
        # collator args
        with open(checkpoint_path / "collator_args.yml", "w") as f:
            f.write(yaml.dump(collator_args.__dict__, Dumper=yaml.Dumper))
        if training_args.push_to_hub:
            push_to_hub(
                training_args.hub_repo,
                checkpoint_path,
                commit_message=f"step {global_step}",
            )
    accelerator.wait_for_everyone()


def train_epoch(epoch):
    global global_step
    eval_results = no_results
    model.train()
    losses = deque(maxlen=training_args.log_every_n_steps)
    break_losses = deque(maxlen=training_args.log_every_n_steps)
    prom_losses = deque(maxlen=training_args.log_every_n_steps)
    step = 0
    # console_rule(f"Epoch {epoch}")
    last_loss = None
    for batch in train_dl:
        with accelerator.accumulate(model):
            if not model_args.use_mpm:
                x = torch.cat(
                    [batch["measures"][m] for m in model_args.measures.split(",")],
                    dim=-1,
                )
            else:
                x = batch["mpm"]
            y = model(x)
            prom_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                y[:, :, 0], batch["prominence"].float(), reduction="none"
            )
            break_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                y[:, :, 1], batch["break"].float(), reduction="none"
            )
            mask_len = batch["mask"].shape[-1]
            prom_loss *= batch["mask"]
            break_loss *= batch["mask"]
            prom_loss = (
                prom_loss.sum()
                * (batch["mask"].sum() / mask_len)
                / prom_loss.shape[-1]
                / prom_loss.shape[0]
            )
            break_loss = (
                break_loss.sum()
                * (batch["mask"].sum() / mask_len)
                / break_loss.shape[-1]
                / break_loss.shape[0]
            )
            loss = (break_loss + prom_loss) / 2
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(
                model.parameters(), training_args.gradient_clip_val
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        losses.append(loss.detach())
        break_losses.append(break_loss.detach())
        prom_losses.append(prom_loss.detach())
        if (
            step > 0
            and step % training_args.log_every_n_steps == 0
            and accelerator.is_main_process
        ):
            last_loss = torch.mean(torch.tensor(losses)).item()
            wandb_log(
                "train",
                {
                    "loss": last_loss,
                    "break_loss": torch.mean(torch.tensor(break_losses)).item(),
                    "prom_loss": torch.mean(torch.tensor(prom_losses)).item(),
                },
                print_log=False,
            )
        if (
            training_args.do_save
            and global_step > 0
            and global_step % training_args.save_every_n_steps == 0
        ):
            save_checkpoint()
        if training_args.n_steps is not None and global_step >= training_args.n_steps:
            return eval_results
        if (
            training_args.eval_every_n_steps is not None
            and global_step > 0
            and global_step % training_args.eval_every_n_steps == 0
            and accelerator.is_main_process
        ):
            eval_results = evaluate()
            console_rule(f"Epoch {epoch}")
        step += 1
        global_step += 1
        if accelerator.is_main_process:
            pbar.update(1)
            if last_loss is not None:
                pbar.set_postfix({"loss": f"{last_loss:.3f}"})

    return eval_results



def evaluate():
    model.eval()
    y_true_prom = []
    y_pred_prom = []
    y_true_break = []
    y_pred_break = []
    losses = []
    prom_losses = []
    break_losses = []
    console_rule("Evaluation Start")
    with torch.no_grad():
        for batch in val_dl:
            if not model_args.use_mpm:
                x = torch.cat(
                    [batch["measures"][m] for m in model_args.measures.split(",")],
                    dim=-1,
                )
            else:
                x = batch["mpm"]
            y = model(x)
            prom_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                y[:, :, 0], batch["prominence"].float(), reduction="none"
            )
            break_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                y[:, :, 1], batch["break"].float(), reduction="none"
            )
            mask_len = batch["mask"].shape[-1]
            prom_loss *= batch["mask"]
            break_loss *= batch["mask"]
            prom_loss = (
                prom_loss.sum() / (batch["mask"].sum() / mask_len) / prom_loss.shape[-1]
            )
            break_loss = (
                break_loss.sum()
                / (batch["mask"].sum() / mask_len)
                / break_loss.shape[-1]
            )
            loss = (break_loss + prom_loss) / 2
            losses.append(loss.detach())
            break_losses.append(break_loss.detach())
            prom_losses.append(prom_loss.detach())
            y_true_prom.append(batch["prominence"][batch["mask"].bool()].flatten())
            y_pred_prom.append(
                torch.sigmoid(y[:, :, 0])[batch["mask"].bool()].flatten()
            )
            y_true_break.append(batch["break"][batch["mask"].bool()].flatten())
            y_pred_break.append(
                torch.sigmoid(y[:, :, 1])[batch["mask"].bool()].flatten()
            )
    # y_true_prom = torch.cat(y_true_prom).cpu().numpy()
    # y_pred_prom = torch.round(torch.cat(y_pred_prom)).cpu().numpy()
    # y_true_break = torch.cat(y_true_break).cpu().numpy()
    # y_pred_break = torch.round(torch.cat(y_pred_break)).cpu().numpy()
    y_true_prom = torch.cat(y_true_prom).cpu().numpy()
    y_pred_prom = torch.cat(y_pred_prom).cpu().numpy()
    y_true_break = torch.cat(y_true_break).cpu().numpy()
    y_pred_break = torch.cat(y_pred_break).cpu().numpy()
    percent_10 = int(len(y_true_prom) * 0.1)
    y_true_prom_t = y_true_prom[:percent_10]
    y_pred_prom_t = y_pred_prom[:percent_10]
    y_true_break_t = y_true_break[:percent_10]
    y_pred_break_t = y_pred_break[:percent_10]
    best_prom_threshold = 0
    best_prom_f1 = 0
    best_break_threshold = 0
    best_break_f1 = 0
    for threshold in np.arange(0, 1, 0.01):
        prom_f1 = f1_score(y_true_prom_t, y_pred_prom_t > threshold, average="binary")
        break_f1 = f1_score(
            y_true_break_t, y_pred_break_t > threshold, average="binary"
        )
        if prom_f1 > best_prom_f1:
            best_prom_f1 = prom_f1
            best_prom_threshold = threshold
        if break_f1 > best_break_f1:
            best_break_f1 = break_f1
            best_break_threshold = threshold
    y_pred_prom = (y_pred_prom > best_prom_threshold)[percent_10:]
    y_pred_break = (y_pred_break > best_break_threshold)[percent_10:]
    y_true_prom = y_true_prom[percent_10:]
    y_true_break = y_true_break[percent_10:]

    eval_results = {
            "loss": torch.mean(torch.tensor(losses)).item(),
            "break_loss": torch.mean(torch.tensor(break_losses)).item(),
            "prom_loss": torch.mean(torch.tensor(prom_losses)).item(),
            "prom_acc": accuracy_score(y_true_prom, y_pred_prom),
            "prom_f1": f1_score(y_true_prom, y_pred_prom, average="macro"),
            "prom_f1_binary": f1_score(y_true_prom, y_pred_prom, average="binary"),
            "prom_precision": precision_score(y_true_prom, y_pred_prom),
            "prom_recall": recall_score(y_true_prom, y_pred_prom),
            "break_acc": accuracy_score(y_true_break, y_pred_break),
            "break_f1": f1_score(y_true_break, y_pred_break, average="macro"),
            "break_f1_binary": f1_score(y_true_break, y_pred_break, average="binary"),
            "break_precision": precision_score(y_true_break, y_pred_break),
            "break_recall": recall_score(y_true_break, y_pred_break),
            "best_prom_threshold": best_prom_threshold,
            "best_break_threshold": best_break_threshold,
            }
    wandb_log(
    "val",
    eval_results,
    )
    return eval_results

def main(multiruns=False):
    global accelerator, training_args, model_args, collator_args, train_dl, val_dl, optimizer, scheduler, model, global_step, pbar

    global_step = 0

    parser = HfArgumentParser([TrainingArgs, BURNModelArgs, CollatorArgs])

    accelerator = Accelerator()

    # parse args
    if len(sys.argv) > 1 and sys.argv[1].endswith(".yml"):
        with open(sys.argv[1], "r") as f:
            args_dict = yaml.load(f, Loader=yaml.Loader)
        # additonally parse args from command line
        (
            training_args,
            model_args,
            collator_args,
        ) = parser.parse_args_into_dataclasses(sys.argv[2:])
        # update args from yaml
        for k, v in args_dict.items():
            used_args = False
            if hasattr(training_args, k):
                setattr(training_args, k, v)
                used_args = True
            if hasattr(model_args, k):
                setattr(model_args, k, v)
                used_args = True
            if hasattr(collator_args, k):
                setattr(collator_args, k, v)
                used_args = True
            if not used_args:
                console_print(f"[yellow]WARNING[/yellow]: unused arg {k} in yaml")
        if len(sys.argv) > 2:
            console_print(
                f"[yellow]WARNING[/yellow]: yaml args will override command line args"
            )
    else:
        (
            training_args,
            model_args,
            collator_args,
        ) = parser.parse_args_into_dataclasses()

    # check if run name is specified
    if training_args.run_name is None:
        raise ValueError("run_name must be specified")
    if (
        training_args.do_save
        and (Path(training_args.checkpoint_path) / training_args.run_name).exists()
    ):
        raise ValueError(f"run_name {training_args.run_name} already exists")

    # wandb
    if accelerator.is_main_process:
        wandb_name, wandb_project, wandb_dir, wandb_mode = (
            training_args.run_name,
            training_args.wandb_project,
            training_args.wandb_dir,
            training_args.wandb_mode,
        )
        wandb_init(wandb_name, wandb_project, wandb_dir, wandb_mode)
        wandb.run.log_code()

    # log args
    console_rule("Arguments")
    console_print(training_args)
    console_print(model_args)
    console_print(collator_args)
    if accelerator.is_main_process:
        wandb_update_config(
            {
                "training": training_args,
                "model": model_args,
                "collator": collator_args,
            }
        )
    collator_args.values_per_word = model_args.values_per_word
    collator_args.measures = model_args.measures

    # Update overwrite specifically if multiruns is true (avoid recomputing features for multiple inits of the probe)
    if multiruns == True:
        training_args.overwrite_data = False
    # Update args from commandine args (training_args)
    model_args.use_mpm = training_args.use_mpm
    model_args.use_cwt = collator_args.use_cwt
    model_args.use_mpm_init = training_args.use_mpm
    model_args.cwt_n_bins = collator_args.cwt_n_bins
    collator_args.overwrite = training_args.overwrite_data
    
    if training_args.use_mpm:
        collator_args.name = collator_args.name.replace("default", "prosody_model")
        bins = training_args.mpm_bin_size
        mask = training_args.mpm_mask_size
        step = training_args.mpm_checkpoint_step
        collator_args.mpm = (
            "cdminix/masked_prosody_model"
            # f"checkpoints/fischer_mpm"
        )

    validate_args(training_args, model_args, collator_args)

    # Distribution Information
    console_rule("Distribution Information")
    console_print(f"[green]accelerator[/green]: {accelerator}")
    console_print(f"[green]n_procs[/green]: {accelerator.num_processes}")
    console_print(f"[green]process_index[/green]: {accelerator.process_index}")

    # model
    model = BreakProminenceClassifier(model_args)
    console_rule("Model")
    print_and_draw_model()

    # dataset
    console_rule("Dataset")

    console_print(f"[green]dataset[/green]: {training_args.burn_dataset}")
    console_print(f"[green]train_split[/green]: {training_args.burn_train_split}")
    console_print(f"[green]val_split[/green]: {training_args.burn_val_split}")

    train_ds = load_dataset(
        training_args.burn_dataset, split=training_args.burn_train_split
    )
    val_ds = load_dataset(
        training_args.burn_dataset, split=training_args.burn_val_split
    )

    console_print(f"[green]train[/green]: {len(train_ds)}")
    console_print(f"[green]val[/green]: {len(val_ds)}")

    collator = get_collator(collator_args)

    # dataloader
    if training_args.num_workers is None:
        train_dl = DataLoader(
            train_ds,
            batch_size=training_args.batch_size,
            shuffle=True,
            collate_fn=collator,
            drop_last=training_args.drop_last,
        )

        val_dl = DataLoader(
            val_ds,
            batch_size=training_args.batch_size,
            shuffle=False,
            collate_fn=collator,
        )
    else:
        train_dl = DataLoader(
            train_ds,
            batch_size=training_args.batch_size,
            shuffle=True,
            collate_fn=collator,
            drop_last=training_args.drop_last,
            num_workers=training_args.num_workers,
        )

        val_dl = DataLoader(
            val_ds,
            batch_size=training_args.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=training_args.num_workers,
        )

    if training_args.overwrite_data:
        console_print(f"[yellow]WARNING[/yellow]: overwriting features")
    if accelerator.is_main_process:
        console_print(f"[green]collator[/green]: doing test run over datasets")
        is_first_batch = True
        for dl in [train_dl, val_dl]:
            for batch in tqdm(dl, total=len(dl)):
                if is_first_batch:
                    if collator_args.name == "default_burn":
                        fig = plot_baseline_batch(batch, collator_args)
                        plt.savefig("figures/first_batch_burn.png")
                    elif collator_args.name == "prosody_model_burn":
                        fig = plot_prosody_model_batch(batch, collator_args)
                        plt.savefig("figures/first_batch_burn.png")
                    wandb.log({"first_batch": wandb.Image(fig)})
                    is_first_batch = False
    collator.args.overwrite = False

    if training_args.num_workers is not None:
        # go back to single accelerator-managed for model
        train_dl = DataLoader(
            train_ds,
            batch_size=training_args.batch_size,
            shuffle=True,
            collate_fn=collator,
            drop_last=training_args.drop_last,
        )
        val_dl = DataLoader(
            val_ds,
            batch_size=training_args.batch_size,
            shuffle=False,
            collate_fn=collator,
        )

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.lr)

    # scheduler
    if training_args.lr_schedule == "linear_with_warmup":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=training_args.lr_warmup_steps,
            num_training_steps=training_args.n_steps,
        )
    else:
        raise NotImplementedError(f"{training_args.lr_schedule} not implemented")

    # accelerator
    model, optimizer, train_dl, val_dl, scheduler = accelerator.prepare(
        model, optimizer, train_dl, val_dl, scheduler
    )

    # evaluation
    if training_args.eval_only:
        console_rule("Evaluation")
        seed_everything(training_args.seed)
        evaluate()
        return

    # training
    console_rule("Training")
    seed_everything(training_args.seed)
    pbar_total = training_args.n_steps
    training_args.n_epochs = training_args.n_steps // len(train_dl) + 1
    console_print(f"[green]n_epochs[/green]: {training_args.n_epochs}")
    console_print(
        f"[green]effective_batch_size[/green]: {training_args.batch_size*accelerator.num_processes}"
    )
    best_results_prom = no_results 
    best_epoch_prom = 0
    best_results_break = no_results 
    best_epoch_break = 0
    pbar = tqdm(total=pbar_total, desc="step")
    for i in range(training_args.n_epochs):
        eval_results = train_epoch(i)
        # Track best epoch
        if eval_results["prom_f1_binary"] > best_results_prom["prom_f1_binary"]:
            best_results_prom = eval_results
            best_epoch_prom = i
        if eval_results["break_f1_binary"] > best_results_break["break_f1_binary"]:
            best_results_break = eval_results
            best_epoch_break = i

    console_rule("Evaluation Start")
    seed_everything(training_args.seed)
    last_results = evaluate()

    # log best results and epoch
    console_rule(f"Best epoch prom {best_epoch_prom}")
    console_rule(f"Best epoch break {best_epoch_break}")
    wandb_log(
        "best_val_results_prom",
        best_results_prom,
    )
    wandb_log(
        "best_val_results_break",
        best_results_break,
    )

    # save final model
    console_rule("Saving")
    if training_args.do_save:
        save_checkpoint()

    # wandb sync reminder
    if accelerator.is_main_process and training_args.wandb_mode == "offline":
        console_rule("Weights & Biases")
        console_print(
            f"use \n[magenta]wandb sync {Path(wandb.run.dir).parent}[/magenta]\nto sync offline run"
        )

    return best_epoch_prom, best_results_prom, best_epoch_break, best_results_break

if __name__ == "__main__":
    # best_epoch_prom, best_results_prom, best_epoch_break, best_results_break = main()
    # print("prominence...")
    # print(best_epoch_prom, best_results_prom)
    # print("break...")
    # print(best_epoch_break, best_results_break)

    # [WIP] couldn't get a bash script/subprocess to run this so quick fix...
    from datetime import datetime
    import pandas as pd
    import numpy as np

    runs=3

    # Collect runs
    best_epochs_prom = {}
    best_results_prom = {}
    best_epochs_break = {}
    best_results_break = {}
    multiruns = False
    for i in range(runs):
        # only overwrite features once for multiruns (i.e., different inits of the probe) 
        if i>0:
            multiruns = True
        best_epoch_prom, best_result_prom, best_epoch_break, best_result_break = main(multiruns=multiruns)
        best_epochs_prom[i] = best_epoch_prom
        best_results_prom[i] = best_result_prom
        best_epochs_break[i] = best_epoch_break
        best_results_break[i] = best_result_break

    # Make writable results
    res_prom_df = pd.DataFrame(best_results_prom).T
    res_prom_df["best_epoch"] = best_epochs_prom.values()
    res_prom_df.loc['mean'] = res_prom_df.mean()
    res_prom_df.loc['std'] = res_prom_df.std()
    print(res_prom_df.mean())

    res_break_df = pd.DataFrame(best_results_break).T
    res_break_df["best_epoch"] = best_epochs_break.values()
    res_break_df.loc['mean'] = res_break_df.mean()
    res_break_df.loc['std'] = res_break_df.std()
    print(res_break_df.mean())

    # Make save file (so janky, didn't want to change the argparsing structure too much though) 
    if collator_args.use_mpm_random:
        model_name = 'MPMrandom'
    elif collator_args.use_cwt:
        model_name = 'CWT'
    elif collator_args.mpm:
        model_name = 'MPM'
    else:
        model_name = 'inputfeatures'
    if 'linear' in sys.argv[1]:
        classifier_name='linear'
    else:
        classifier_name='conformer'
    
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"results/burn_prominence/{classifier_name}_{model_name}_{current_datetime}.json"
    print(f"Saving to: {filename}")
    res_prom_df.to_json(filename)
    filename = f"results/burn_break/{classifier_name}_{model_name}_{current_datetime}.json"
    print(f"Saving to: {filename}")
    res_break_df.to_json(filename)
