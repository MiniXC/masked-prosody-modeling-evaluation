[![tests](https://github.com/MiniXC/masked-prosody-modeling/actions/workflows/run_lint_and_test.yml/badge.svg)](https://github.com/MiniXC/ml-template/actions/workflows/run_lint_and_test.yml)
# ml-template
Template for my machine learning projects. 

It's quite opinionated in the following (and probably more) ways:
- uses accelerate
- splits up config, model and scripts
- assumes one is always using a huggingface dataset (this can also be done using ``load_dataset('some_dataset.py')``)
- uses collators as the main way to process data that hasn't been preprocessed by the dataset
- uses separate configs for training (everything not shipped with the model), model and collator

## architecture
The following updates automatically every time one of the training scripts is run.

## BURN (Boston University Radio News)
<details>
<summary>Click to expand</summary>
<img src="./figures/model_burn.gv.png"></img>
</details>

## RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
<details>
<summary>Click to expand</summary>
<img src="./figures/model_ravdess.gv.png"></img>
</details>

## TIMIT (Acoustic-Phonetic Continuous Speech Corpus)
<details>
<summary>Click to expand</summary>
<img src="./figures/model_timit.gv.png"></img>
</details>

## first training batch
<details>
<summary>Click to expand</summary>
<img src="./figures/first_batch.png"></img>
</details>