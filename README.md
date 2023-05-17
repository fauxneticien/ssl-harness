# ssl-harness

A lightweight test harness for evaluating self-supervised models for speech (mainly HuBERT).

# Usage

## Download data

Download LibriSpeech / LibriLight data for development.

```python
python download-libris.py
```

## Download model checkpoint (if fine-tuning)

```bash
gdown 10UHFElbsSZaQmQilfBtaDXzbD9QmKRGa -O tmp/
```

## Run training

### Fine-tune model

```bash
python finetune.py \
  --dataset-path ./data/ \
  --exp-dir ./tmp/exp_finetune2 \
  --checkpoint ./tmp/hubert_iter2_checkpoint.pt \
  --subset 1h \
  --gpus 2 \
  --debug \
  --warmup-updates 2000 \
  --hold-updates 8000 \
  --decay-updates 10000 \
  --max-updates 20000 \
  --learning-rate 5e-5
```
