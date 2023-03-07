# Evaluation with JGLUE

[JGLUE](https://github.com/yahoojapan/JGLUE) is a Japanese NLP task set.

This folder contains the resources for the evaluation of chiTra models on JGLUE.

## Results

The performance on the JGLUE dev set are shown in below table.
Results for other models are taken from [JGLUE - Baseline Score](https://github.com/yahoojapan/JGLUE#baseline-scores).

| Model                       | MARC-ja | JSTS             | JNLI  | JSQuAD      | JCommonsenseQA |
| --------------------------- | ------- | ---------------- | ----- | ----------- | -------------- |
|                             | acc     | Pearson/Spearman | acc   | EM/F1       | acc            |
| chiTra-1.0                  | 0.956   | 0.903/0.861      | 0.882 | 0.839/0.919 | 0.788          |
| chiTra-1.1                  | 0.960   | 0.916/0.876      | 0.900 | 0.860/0.937 | 0.840          |
|                             |         |
| Tohoku BERT base            | 0.958   | 0.909/0.868      | 0.899 | 0.871/0.941 | 0.808          |
| Tohoku BERT base (char)     | 0.956   | 0.893/0.851      | 0.892 | 0.864/0.937 | 0.718          |
| Tohoku BERT large           | 0.955   | 0.913/0.872      | 0.900 | 0.880/0.946 | 0.816          |
| NICT BERT base              | 0.958   | 0.910/0.871      | 0.902 | 0.897/0.947 | 0.823          |
| Waseda RoBERTa base         | 0.962   | 0.913/0.873      | 0.895 | 0.864/0.927 | 0.840          |
| Waseda RoBERTa large (s128) | 0.954   | 0.930/0.896      | 0.924 | 0.884/0.940 | 0.907          |
| Waseda RoBERTa large (s512) | 0.961   | 0.926/0.892      | 0.926 | 0.918/0.963 | 0.891          |
| XLM RoBERTa base            | 0.961   | 0.877/0.831      | 0.893 | -/-         | 0.687          |
| XLM RoBERTa large           | 0.964   | 0.918/0.884      | 0.919 | -/-         | 0.840          |

Note that chiTra-1.0 and 1.1 are base-size BERT model.
Comparing to the base models, chiTra-1.1 achieves comparable results.

In JSQuAD task, chiTra performs a little poorly.
Due to the normalization feature of the chiTra tokenizer and the difficulty of
taking alignment of the original and normalized texts, chiTra model outputs normalized text as a result.
In some cases this cause a failure in the answer matching.

## Reproduction

We provide a patch file to modify [the transformers library](https://github.com/huggingface/transformers).
You can follow the instructions in [JGLUE fine-turning page](https://github.com/yahoojapan/JGLUE/tree/main/fine-tuning), replacing the patch file with ours.
You will also need to install some python modules (use `requirements.txt`).
We used `transformers-v4.26.1` to generate our patch, but other versions may work.

The main additions of our patch comparing to JGLUE's are followings:

- Assign `sudachitra.BertSudachipyTokenizer` to `transformers.AutoTokenizer`, so that we can auto-load chiTra tokenizer.
- Modification for JSQuAD task:
  - Pretokenize datasets using Sudachi tokenizer, instead of whitespace-separation.
  - Manage the alignment gap caused by the normalization feature of the chiTra tokenizer.
  - Remove whitespaces from the final output of the model in the evaluation.
