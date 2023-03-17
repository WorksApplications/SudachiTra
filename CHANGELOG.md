# chiTra-1.1 model (2023-03-17)

- A pretrained Japanese BERT base model, trained using chiTra tokenizer.

## Updates / Changes

- Cleaning processes of the NWJC corpus are added.
  - Total size after cleaning is 79 GB.
- Vocabulary is rebuilt in the same way.
  - Total vocab size is `32597`.
- Sudachi libraries are updated to:
  - SudachiPy: `0.6.6`.
  - SudachiDict: `20220729-core`.
  - SudachiTra: `0.1.8`.
- `word_form_type` is changed to `normalized_nouns`.
- Total training steps is increased to `20472`.

# [0.1.8](https://github.com/WorksApplications/SudachiTra/releases/tag/v0.1.8) (2023-03-10)

## Highlights

- Add new `word_format_type`: `normalized_nouns`. (#48, #50)
  - Normalizes morphemes that do not have conjugation form.

## Other

- Faster part-of-speech matching (#36)
- Use HuggingFace compatible pretokenizer (#38)
- Fix/Update pretraining scripts and documents (#39, #40, #45, #46)
- Fix github test workflow (#49)
- Enable to save vocab file with duplicated items (#54)

# chiTra-1.0 (2022-02-25)

- A pretrained Japanese BERT base model, trained using chiTra tokenizer.

## Details

- Model
  - chiTra-1.0 is a BERT base model.
- Corpus
  - We used NINJAL Web Japanese Corpus (NWJC) from National Institute for Japanese Language and Linguistics.
  - Cleaning process is explained [here](https://github.com/WorksApplications/SudachiTra/tree/main/pretraining/bert#2-preprocessing-corpus-cleaning).
    - Total size after cleaning is 109 GB.
- Vocabulary
  - Vocabulary is built on the above corpus, [using WordPiece](https://github.com/WorksApplications/SudachiTra/tree/main/pretraining/bert#wordpiece) and vocab size 32000.
  - We added 常用漢字 and 人名用漢字 to cover usual Japanese text.
    - Total vocab size is `32615`.
- Sudachi libraries
  - SudachiPy: `0.6.2`
  - SudachiDict: `20211220-core`
  - chiTra: `0.1.7`
    - We used `word_form_type`: `normalized_and_surface`.
- Training Parameters
  - See [our paper](https://github.com/WorksApplications/SudachiTra#chitra%E3%81%AE%E5%BC%95%E7%94%A8--citing-chitra)) or [pretraining page](https://github.com/WorksApplications/SudachiTra/tree/main/pretraining/bert#5training).
  - Total training step is `10236`.
