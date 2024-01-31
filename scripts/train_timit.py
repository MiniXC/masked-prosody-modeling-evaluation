import os
import sys
from collections import deque
from pathlib import Path

sys.path.append(".")  # add root of project to path

# torch & hf
import torchvision
import torch
from torch.utils.data import DataLoader
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
import matplotlib.pyplot as plt
from rich.console import Console

console = Console()

# local imports
from configs.args import (
    TrainingArgs,
    TIMITModelArgs as ModelArgs,
    TIMITCollatorArgs as CollatorArgs,
)
from configs.validation import validate_args
from util.remote import wandb_update_config, wandb_init, push_to_hub
from util.plotting import plot_baseline_timit_batch as plot_baseline_batch
from util.plotting import plot_prosody_model_timit_batch as plot_prosody_batch
from model.timit_classifiers import PhonemeWordBoundaryClassifier
from collators import get_collator

no_results = {
        "loss": 100000,
        "phon_loss": 100000,
        "word_loss": 100000,
        "phon_f1": 0,
        "phon_precision": 0,
        "phon_recall": 0,
        "phon_accuracy": 0,
        "word_f1": 0,
        "word_precision": 0,
        "word_recall": 0,
        "word_accuracy": 0,
        "phon_threshold": 0,
        "word_threshold": 0,
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
            filename="model_timit",
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
    phon_losses = deque(maxlen=training_args.log_every_n_steps)
    word_losses = deque(maxlen=training_args.log_every_n_steps)
    step = 0
    # console_rule(f"Epoch {epoch}")
    last_loss = None
    for batch in train_dl:
        with accelerator.accumulate(model):
            if not training_args.use_mpm:
                x = torch.cat(
                    [batch["measures"][m] for m in model_args.measures.split(",")],
                    dim=-1,
                )
            else:
                x = batch["mpm"]
            y = model(x)
            phon_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                y[:, :, 0],
                batch["phoneme_boundaries"].float(),
                reduction="none",
            )
            word_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                y[:, :, 1],
                batch["word_boundaries"].float(),
                reduction="none",
            )
            mask_len = batch["mask"].shape[-1]
            phon_loss *= batch["mask"]
            word_loss *= batch["mask"]
            phon_loss = (
                phon_loss.sum() * (batch["mask"].sum() / mask_len) / phon_loss.shape[-1]
            )
            word_loss = (
                word_loss.sum() * (batch["mask"].sum() / mask_len) / word_loss.shape[-1]
            )
            loss = (phon_loss + word_loss) / 3
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(
                model.parameters(), training_args.gradient_clip_val
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        losses.append(loss.detach())
        phon_losses.append(phon_loss.detach())
        word_losses.append(word_loss.detach())
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
                    "phon_loss": torch.mean(torch.tensor(phon_losses)).item(),
                    "word_loss": torch.mean(torch.tensor(word_losses)).item(),
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
            if training_args.do_full_eval:
                eval_results = evaluate()
            else:
                evaluate_loss_only()
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
    y_true_phon = []
    y_pred_phon = []
    y_true_word = []
    y_pred_word = []
    losses = []
    phon_losses = []
    word_losses = []
    console_rule("Evaluation Start")
    with torch.no_grad():
        for batch in val_dl:
            if not training_args.use_mpm:
                x = torch.cat(
                    [batch["measures"][m] for m in model_args.measures.split(",")],
                    dim=-1,
                )
            else:
                x = batch["mpm"]
            y = model(x)
            phon_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                y[:, :, 0],
                batch["phoneme_boundaries"].float(),
                reduction="none",
            )
            word_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                y[:, :, 1],
                batch["word_boundaries"].float(),
                reduction="none",
            )
            mask_len = batch["mask"].shape[-1]
            phon_loss *= batch["mask"]
            word_loss *= batch["mask"]
            phon_loss = (
                phon_loss.sum() * (batch["mask"].sum() / mask_len) / phon_loss.shape[-1]
            )
            word_loss = (
                word_loss.sum() * (batch["mask"].sum() / mask_len) / word_loss.shape[-1]
            )
            loss = (phon_loss + word_loss) / 3
            losses.append(loss.detach())
            phon_losses.append(phon_loss.detach())
            word_losses.append(word_loss.detach())
            y_true_phon.append(
                batch["phoneme_boundaries"][batch["mask"] == 1].flatten()
            )
            y_pred_phon.append(torch.sigmoid(y[:, :, 0])[batch["mask"] == 1].flatten())
            y_true_word.append(batch["word_boundaries"][batch["mask"] == 1].flatten())
            y_pred_word.append(torch.sigmoid(y[:, :, 1])[batch["mask"] == 1].flatten())
    # get threshold using 10% of data
    y_true_phon = torch.cat(y_true_phon).cpu().numpy()
    y_pred_phon = torch.cat(y_pred_phon).cpu().numpy()
    y_true_word = torch.cat(y_true_word).cpu().numpy()
    y_pred_word = torch.cat(y_pred_word).cpu().numpy()
    percent_10 = int(len(y_true_phon) * 0.1)
    y_true_phon_t = y_true_phon[:percent_10]
    y_pred_phon_t = y_pred_phon[:percent_10]
    y_true_word_t = y_true_word[:percent_10]
    y_pred_word_t = y_pred_word[:percent_10]
    best_phon_threshold = 0
    best_word_threshold = 0
    best_phon_f1 = 0
    best_word_f1 = 0
    for threshold in np.arange(0, 1, 0.01):
        phon_f1 = f1_score(y_true_phon_t, y_pred_phon_t > threshold)
        word_f1 = f1_score(y_true_word_t, y_pred_word_t > threshold)
        if phon_f1 > best_phon_f1:
            best_phon_f1 = phon_f1
            best_phon_threshold = threshold
        if word_f1 > best_word_f1:
            best_word_f1 = word_f1
            best_word_threshold = threshold
    y_pred_phon = (y_pred_phon > best_phon_threshold)[percent_10:]
    y_pred_word = (y_pred_word > best_word_threshold)[percent_10:]
    y_true_phon = y_true_phon[percent_10:]
    y_true_word = y_true_word[percent_10:]
    console_print(f"[green]pct. positive phoneme[/green]: {y_true_phon.mean():.3f}")
    console_print(f"[green]pct. positive word[/green]: {y_true_word.mean():.3f}")

    eval_results = {
            "loss": torch.mean(torch.tensor(losses)).item(),
            "phon_loss": torch.mean(torch.tensor(phon_losses)).item(),
            "word_loss": torch.mean(torch.tensor(word_losses)).item(),
            "phon_f1": f1_score(y_true_phon, y_pred_phon),
            "phon_precision": precision_score(y_true_phon, y_pred_phon),
            "phon_recall": recall_score(y_true_phon, y_pred_phon),
            "phon_accuracy": accuracy_score(y_true_phon, y_pred_phon),
            "word_f1": f1_score(y_true_word, y_pred_word),
            "word_precision": precision_score(y_true_word, y_pred_word),
            "word_recall": recall_score(y_true_word, y_pred_word),
            "word_accuracy": accuracy_score(y_true_word, y_pred_word),
            "phon_threshold": best_phon_threshold,
            "word_threshold": best_word_threshold,
                    }
    wandb_log(
        "val",
        eval_results,
        )

    return eval_results


def evaluate_loss_only():
    model.eval()
    losses = []
    phon_losses = []
    word_losses = []
    console_rule("Evaluation")
    with torch.no_grad():
        for batch in val_dl:
            if not training_args.use_mpm:
                x = torch.cat(
                    [batch["measures"][m] for m in model_args.measures.split(",")],
                    dim=-1,
                )
            else:
                x = batch["mpm"]
            y = model(x)
            phon_loss = torchvision.ops.focal_loss.sigmoid_focal_loss(
                y[:, :, 0],
                batch["phoneme_boundaries"].float(),
                reduction="none",
                alpha=training_args.timit_phon_focal_loss_alpha,
            )
            word_loss = (
                torchvision.ops.focal_loss.sigmoid_focal_loss(
                    y[:, :, 1],
                    batch["word_boundaries"].float(),
                    reduction="none",
                    alpha=training_args.timit_word_focal_loss_alpha,
                )
                * 2
            )
            mask_len = batch["mask"].shape[-1]
            phon_loss *= batch["mask"]
            word_loss *= batch["mask"]
            phon_loss = (
                phon_loss.sum() * (batch["mask"].sum() / mask_len) / phon_loss.shape[-1]
            )
            word_loss = (
                word_loss.sum() * (batch["mask"].sum() / mask_len) / word_loss.shape[-1]
            )
            loss = (phon_loss + word_loss) / 3
            losses.append(loss.detach())
            phon_losses.append(phon_loss.detach())
            word_losses.append(word_loss.detach())
    wandb_log(
        "val",
        {
            "loss": torch.mean(torch.tensor(losses)).item(),
            "phon_loss": torch.mean(torch.tensor(phon_losses)).item(),
            "word_loss": torch.mean(torch.tensor(word_losses)).item(),
        },
    )


def main():
    global accelerator, training_args, model_args, collator_args, train_dl, val_dl, optimizer, scheduler, model, global_step, pbar

    global_step = 0

    parser = HfArgumentParser([TrainingArgs, ModelArgs, CollatorArgs])

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
            if hasattr(training_args, k):
                setattr(training_args, k, v)
            if hasattr(model_args, k):
                setattr(model_args, k, v)
            if hasattr(collator_args, k):
                setattr(collator_args, k, v)
        if len(sys.argv) > 2:
            console_print(
                f"[yellow]WARNING[/yellow]: yaml args will be override command line args"
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
    collator_args.measures = model_args.measures
    model_args.use_mpm = training_args.use_mpm
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
    model = PhonemeWordBoundaryClassifier(model_args)
    console_rule("Model")
    print_and_draw_model()

    # dataset
    console_rule("Dataset")

    console_print(f"[green]dataset[/green]: {training_args.timit_dataset}")
    console_print(f"[green]train_split[/green]: {training_args.timit_train_split}")
    console_print(f"[green]val_split[/green]: {training_args.timit_val_split}")

    train_ds = load_dataset(
        training_args.timit_dataset,
        split=training_args.timit_train_split,
        data_dir=os.environ["TIMIT_PATH"],
    )
    val_ds = load_dataset(
        training_args.timit_dataset,
        split=training_args.timit_val_split,
        data_dir=os.environ["TIMIT_PATH"],
    )

    console_print(f"[green]train[/green]: {len(train_ds)}")
    console_print(f"[green]val[/green]: {len(val_ds)}")

    collator = get_collator(collator_args, device=accelerator.device)

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
                    if collator_args.name == "default_timit":
                        fig = plot_baseline_batch(batch, collator_args)
                    elif collator_args.name == "prosody_model_timit":
                        fig = plot_prosody_batch(batch, collator_args)
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
        eval_results = evaluate()
        return eval_results

    # training
    console_rule("Training")
    seed_everything(training_args.seed)
    pbar_total = training_args.n_steps
    training_args.n_epochs = training_args.n_steps // len(train_dl) + 1
    console_print(f"[green]n_epochs[/green]: {training_args.n_epochs}")
    console_print(
        f"[green]effective_batch_size[/green]: {training_args.batch_size*accelerator.num_processes}"
    )
    # Track results
    best_results = no_results 
    best_epoch = 0
    pbar = tqdm(total=pbar_total, desc="step")
    for i in range(training_args.n_epochs):
        eval_results = train_epoch(i) 
        # Track best epoch
        if eval_results["phon_f1"] > best_results["phon_f1"]:
            best_results = eval_results
            best_epoch = i
    console_rule("Evaluation Start")
    seed_everything(training_args.seed)
    last_results = evaluate()

    # log best results and epoch
    console_rule(f"Best epoch {best_epoch}")
    wandb_log(
        "best_val_results",
        best_results,
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

    return best_epoch, best_results

if __name__ == "__main__":
    # best_epoch, best_result = main()

    # [WIP] couldn't get a bash script/subprocess to run this so quick fix...
    from datetime import datetime
    import pandas as pd
    import numpy as np

    runs=3

    # Collect runs
    best_epochs = {}
    best_results = {}
    for i in range(runs):
        best_epoch, best_result = main()
        best_epochs[i] = best_epoch
        best_results[i] = best_result

    # Make writable results
    res_df = pd.DataFrame(best_results).T
    res_df["best_epoch"] = best_epochs.values()
    res_df.loc['mean'] = res_df.mean()
    res_df.loc['std'] = res_df.std()
    print(res_df.mean())

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
    filename = f"results/timit/{classifier_name}_{model_name}_{current_datetime}.json"
    res_df.to_json(filename)