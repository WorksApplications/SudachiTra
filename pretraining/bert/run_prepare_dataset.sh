#!/bin/bash

cd $(dirname $0)

DATASET_DIR="./datasets"
CORPUS_DIR="${DATASET_DIR}/corpus"
SPLITTED_CORPUS_DIR="${DATASET_DIR}/corpus_splitted_by_paragraph"

# download dataset
mkdir -p ${CORPUS_DIR}
for target in "train" "validation" "test"; do
  time python3 prepare_dataset.py --target ${target} > ${CORPUS_DIR}/ja_wiki40b_${target}.txt
done

### split dataset for each paragraph

mkdir -p ${SPLITTED_CORPUS_DIR}
for target in "train" "validation" "test"; do
  cat ${CORPUS_DIR}/ja_wiki40b_${target}.txt | sed -e "s/_START_ARTICLE_//g" -e "s/_START_PARAGRAPH_//g" | cat -s > ${SPLITTED_CORPUS_DIR}/ja_wiki40b_${target}.paragraph.txt
done

