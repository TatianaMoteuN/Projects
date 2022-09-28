#!/usr/bin/env bash 
# -*- coding: utf-8 -*- 



# author: xiaoy li 
# description:
# clean code and add comments 




REPO_PATH = /content/drive/My\ Drive/seq2seq_coref/CorefQA
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"
export TPU_NAME=tf-tpu
export TPU_ZONE=europe-west4-a
export GCP_PROJECT=xiaoyli-20-01-4820

BERT_DIR=gs://content/drive/My\ Drive/seq2seq_coref/bart.large
DATA_DIR=gs://content/drive/My\ Drive/seq2seq_coref/doc
OUTPUT_DIR=gs://content/drive/My\ Drive/seq2seq_coref/save_model

python3 ${REPO_PATH}/run/run_corefqa.py \
--output_dir=$OUTPUT_DIR \
--bert_config_file=$BERT_DIR/encoder.json \
--init_checkpoint=$BERT_DIR/model.pt \
--vocab_file=$BERT_DIR/vocab.bpe \
--logfile_path=$OUTPUT_DIR/eval.log \
--num_epochs=8 \
--keep_checkpoint_max=50 \
--save_checkpoints_steps=500 \
--train_file=$DATA_DIR/train.corefqa.english.tfrecord \
--dev_file=$DATA_DIR/dev.corefqa.english.tfrecord \
--test_file=$DATA_DIR/test.corefqa.english.tfrecord \
--do_train=False \
--do_eval=True \
--do_predict=True \
--learning_rate=8e-4 \
--dropout_rate=0.2 \
--mention_threshold=0.5 \
--hidden_size=1024 \
--num_docs=5604 \
--window_size=384 \
--num_window=6 \
--max_num_mention=50 \
--start_end_share=False \
--max_span_width=10 \
--max_candiate_mentions=100 \
--top_span_ratio=0.2 \
--max_top_antecedents=30 \
--max_query_len=150 \
--max_context_len=150 \
--sec_qa_mention_score=False \
--use_tpu=True \
--tpu_name=$TPU_NAME \
--tpu_zone=$TPU_ZONE \
--gcp_project=$GCP_PROJECT \
--num_tpu_cores=1 \
--seed=2333