#!/bin/bash
set -eu

# set your own dir
SCRIPT_DIR="./scripts"
MODEL_ROOT="./bert"
DATASET_ROOT="./datasets"
OUTPUT_ROOT="./out"

# model to search
MODEL_NAMES=(
  "tohoku"
  "kyoto"
  "nict"
  "chitra_surface"
  "chitra_normalized_and_surface"
  "chitra_normalized_conjugation"
  "chitra_normalized"
)

DATASETS=("amazon" "rcqa" "kuci")

# Hyperparameters from Appendix A.3, Devlin et al., 2019
BATCHES=(16 32)
LRS=(5e-5 3e-5 2e-5)
EPOCHS=(2 3 4)

# set path to the model files
declare -A MODEL_DIRS=(
  ["tohoku"]="cl-tohoku/bert-base-japanese-whole-word-masking"
  ["kyoto"]="${MODEL_ROOT}/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers"
  ["nict"]="${MODEL_ROOT}/NICT_BERT-base_JapaneseWikipedia_32K_BPE"
  ["chitra_surface"]="${MODEL_ROOT}/Wikipedia_surface/phase_2"
  ["chitra_normalized_and_surface"]="${MODEL_ROOT}/Wikipedia_normalized_and_surface/phase_2"
  ["chitra_normalized_conjugation"]="${MODEL_ROOT}/Wikipedia_normalized_conjugation/phase_2"
  ["chitra_normalized"]="${MODEL_ROOT}/Wikipedia_normalized/phase_2"
)

function set_model_args() {
  MODEL=$1
  DATASET=$2
  MODEL_DIR="${MODEL_DIRS[$1]}"
  DATASET_DIR="${DATASET_ROOT}/${DATASET}"
  OUTPUT_DIR="${OUTPUT_ROOT}/${MODEL}_${DATASET}/${LR}_${BATCH}_${EPOCH}/"
  export MODEL DATASET MODEL_DIR DATASET_DIR OUTPUT_DIR

  # pretokenizer
  PRETOKENIZER="identity"
  if [ ${MODEL} = "kyoto" ] ; then
    PRETOKENIZER="juman"
  elif [ ${MODEL} = "nict" ] ; then
    PRETOKENIZER="mecab-juman"
  fi
  export PRETOKENIZER

  # tokenizer (sudachi)
  TOKENIZER=${MODEL_DIR}
  if [ ${MODEL:0:6} = "chitra" ] ; then
    TOKENIZER="sudachi"
  fi
  export TOKENIZER
}

command_echo='( echo \
  "${MODEL}, ${DATASET}, ${MODEL_DIR}, ${DATASET_DIR}, ${OUTPUT_DIR}, " \
  "${PRETOKENIZER}, ${TOKENIZER}, ${BATCH}, ${LR}, ${EPOCH}, " \
)'

export SCRIPT_PATH="${SCRIPT_DIR}/run_evaluation.py"
command_run='( \
  python ${SCRIPT_PATH} \
    --model_name_or_path          ${MODEL_DIR} \
    --pretokenizer_name           ${PRETOKENIZER} \
    --tokenizer_name              ${TOKENIZER} \
    --dataset_name                ${DATASET} \
    --dataset_dir                 ${DATASET_DIR} \
    --output_dir                  ${OUTPUT_DIR} \
    --do_train                    \
    --do_eval                     \
    --do_predict                  \
    --gradient_accumulation_steps $((BATCH / 8)) \
    --per_device_eval_batch_size  64 \
    --per_device_train_batch_size 8 \
    --learning_rate               ${LR} \
    --num_train_epochs            ${EPOCH} \
    --overwrite_cache \
    # --max_train_samples           100 \
    # --max_val_samples             100 \
    # --max_test_samples            100 \
)'

# mkdir for log
mkdir -p logs
/bin/true > logs/jobs.txt

for DATASET in ${DATASETS[@]}; do
  for MODEL in ${MODEL_NAMES[@]}; do
    for BATCH in ${BATCHES[@]}; do
      for LR in ${LRS[@]}; do
        for EPOCH in ${EPOCHS[@]}; do
          export BATCH LR EPOCH
          set_model_args ${MODEL} ${DATASET}

          script -c "${command_echo}" logs/echo.log
          script -c "${command_run}" logs/${MODEL}_${DATASET}_batch${BATCH}_lr${LR}_epochs${EPOCH}.log
        done
      done
    done
  done
done
