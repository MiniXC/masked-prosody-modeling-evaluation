from itertools import product

bin_sizes = [4, 8, 16, 32, 64, 128, 512, 1024]
mask_sizes = [1, 2, 4, 8, 16, 32, 64, 128, "_random"]
combinations = list(product(bin_sizes, mask_sizes))
machines = {
    "starariel-0": [],
    "starariel-1": [],
    "starariel-2": [],
    "starariel-3": [],
    "hessdalen-0": [],
    "hessdalen-1": [],
    "hessdalen-2": [],
    "hessdalen-3": [],
}
# divide the combinations into 8 groups

for i, machine in enumerate(machines):
    machines[machine] = combinations[i::8]

with open("scripts/train.sh", "w") as f:
    f.write("#!/bin/bash\n")
    f.write("if [ \"$1\" == \"--setup\" ]; then\n")
    f.write("\tcd /disk/scratch/s1764494\n")
    f.write("\tgcloud storage cp gs://datasets-cdminix/timit.zip .\n")
    f.write("\tunzip timit.zip\n")
    f.write("\tmv timit data/timit_0\n")
    f.write("\tcp -r data/timit_0 data/timit_1\n")
    f.write("\tcp -r data/timit_0 data/timit_2\n")
    f.write("\tcp -r data/timit_0 data/timit_3\n")
    f.write("\tgcloud storage cp gs://datasets-cdminix/bu_radio.zip .\n")
    f.write("\tunzip bu_radio.zip\n")
    f.write("\tmv bu_radio data/bu_radio_0\n")
    f.write("\tcp -r data/bu_radio_0 data/bu_radio_1\n")
    f.write("\tcp -r data/bu_radio_0 data/bu_radio_2\n")
    f.write("\tcp -r data/bu_radio_0 data/bu_radio_3\n")
    f.write("\tcd /disk/scratch/s1764494/masked-prosody-modeling-evaluation\n")
    f.write("\tgcloud storage cp -R gs://masked-prosody-model/checkpoints .\n")
    f.write("\tmv checkpoints mpm_checkpoints\n")
    f.write("fi\n")
    for machine, combs in machines.items():
        # use --machine to specify the machine
        f.write('if [ "$1" == "--machine" ] && [ "$2" == "{}" ]; then\n'.format(machine))
        gpu_num = machine.split("-")[1]
        burn_preamble = f'\tCUDA_VISIBLE_DEVICES={gpu_num} BURN_PATH="/disk/scratch/s1764494/data/bu_radio_{gpu_num}" HF_DATASETS_CACHE="/disk/scratch/s1764494/data/hf_{gpu_num}" '
        timit_preamble = f'\tCUDA_VISIBLE_DEVICES={gpu_num} TIMIT_PATH="/disk/scratch/s1764494/data/timit_{gpu_num}" HF_DATASETS_CACHE="/disk/scratch/s1764494/data/hf_{gpu_num}" '
        ravdess_preamble = f'\tCUDA_VISIBLE_DEVICES={gpu_num} HF_DATASETS_CACHE="/disk/scratch/s1764494/data/hf_{gpu_num}" '
        linear_command = 'python scripts/train_{dataset}.py configs/mpm_linear.yml --run_name {dataset}_bin{bin}_mask{mask}_linear_{num} --mpm_bin_size {bin} --mpm_mask_size {mask} --use_mpm --mpm_layer 7'
        conformer_command = 'python scripts/train_{dataset}.py configs/mpm_conformer.yml --run_name {dataset}_bin{bin}_mask{mask}_conformer --mpm_bin_size {bin} --mpm_mask_size {mask} --use_mpm --mpm_layer 7'
        for bin_size, mask_size in combs:
            for dataset in ["burn", "timit", "ravdess"]:
                if dataset == "burn":
                    preamble = burn_preamble
                elif dataset == "timit":
                    preamble = timit_preamble
                elif dataset == "ravdess":
                    preamble = ravdess_preamble
                f.write(preamble + linear_command.format(bin=bin_size, mask=mask_size, dataset=dataset, num=1) + " --overwrite_data\n")
                f.write(preamble + linear_command.format(bin=bin_size, mask=mask_size, dataset=dataset, num=2) + "\n")
                f.write(preamble + linear_command.format(bin=bin_size, mask=mask_size, dataset=dataset, num=3) + "\n")
                f.write(preamble + conformer_command.format(bin=bin_size, mask=mask_size, dataset=dataset) + "\n")
        f.write("fi\n")