# Evaluation

This document describes about our evaluation experiments.

Check `README.md` for detailed usage of each scripts.


## Performance with downstream tasks

Evaluate model with 3 tasks.

### Tasks

- The Multilingual Amazon Reviews Corpus (Amazon)
    - https://registry.opendata.aws/amazon-reviews-ml/
    - Text classification task / 文章分類
- 京都大学常識推論データセット (KUCI)
    - https://nlp.ist.i.kyoto-u.ac.jp/?KUCI
    - Multiple choice task / 常識推論
- 解答可能性付き読解データセット (RCQA)
    - http://www.cl.ecei.tohoku.ac.jp/rcqa/
    - Question answering task (SQuAD2.0 format) / 読解

### Steps

example for Amazon task:

```bash
# Generate dataset for evaluation.
# Download raw data first for KUCI and RCQA.
python convert_dataset.py amazon --output ./datasets/amazon

# Run finetuning/prediction with hyper parameter search.
# Place model files under `./bert/`.
./run_all.sh amazon

# Correct test result file (for chitra surface model).
python summary_results.py amazon ./out/chitra_surface_amazon/ --output ./summary.csv
```


## Robustness to the text normalization

Run evaluation with test data whose texts are normalized.

Ideal model should be robust to this change (outputs remain same after nomralization).

### Steps

```bash
# Generate normalized dataset.
python convert_dataset.py amazon --output ./datasets_normalized/amazon

# Following steps are same to the model evaluation, but need to modify
# dataset dir name in `run_all.sh` to `datasets_normalized` in this case.
```

