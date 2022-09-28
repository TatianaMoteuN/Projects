#!/bin/bash

DATASET=$1
echo "processing $1"

cd /content/drive/My\ Drive/seq2seq_coref/fairseq
#cd ..

# BPE preprocessing.
for SPLIT in train dev; do
    for LANG in "source" "target"; do
        python -m examples.roberta.multiprocessing_bpe_encoder\
            --encoder-json /content/drive/My\ Drive/seq2seq_coref/bart.large/encoder.json\
            --vocab-bpe /content/drive/My\ Drive/seq2seq_coref/bart.large/vocab.bpe \
            --inputs "$DATASET/$SPLIT.$LANG" \
            --outputs "$DATASET/$SPLIT.bpe.$LANG" \
            --workers 60 \
            --keep-empty;
    done
done

cd ..

# Binarize the dataset.
fairseq-preprocess --source-lang "source" --target-lang "target" \
    --trainpref "$DATASET/train.bpe" \
    --validpref "$DATASET/dev.bpe" \
    --destdir "$DATASET/bin" \
    --workers 60 \
    --srcdict /content/drive/My\ Drive/seq2seq_coref/bart.large/dict.txt \
    --tgtdict /content/drive/My\ Drive/seq2seq_coref/bart.large/dict.txt;

DATASET=$1
echo "processing $1"

cd /content/drive/My\ Drive/seq2seq_coref/fairseq

# BPE preprocessing.
for SPLIT in train dev; do
    for LANG in "source" "target"; do
        python -m examples.roberta.multiprocessing_bpe_encoder\
            --encoder-json /content/drive/My\ Drive/seq2seq_coref/bart.large/encoder.json\
            --vocab-bpe /content/drive/My\ Drive/seq2seq_coref/bart.large/vocab.bpe \
            --inputs "$DATASET/$SPLIT.$LANG" \
            --outputs "$DATASET/$SPLIT.bpe.$LANG" \
            --workers 60 \
            --keep-empty;
    done
done
cd ..
# Binarize the dataset.
fairseq-preprocess --source-lang "source" --target-lang "target" \
    --trainpref "$DATASET/train.bpe" \
    --validpref "$DATASET/dev.bpe" \
    --destdir "$DATASET/bin" \
    --workers 60 \
    --srcdict /content/drive/My\ Drive/seq2seq_coref/bart.large/dict.txt \
    --tgtdict /content/drive/My\ Drive/seq2seq_coref/bart.large/dict.txt;
