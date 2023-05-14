# ssl-harness

A lightweight test harness for evaluating self-supervised models for speech (mainly HuBERT).

## Motivation

Pre-training code for wav2vec 2.0 and HuBERT is tightly integrated into the (sprawling) [fairseq](https://github.com/facebookresearch/fairseq) library.
Implementations outside of fairseq are similarly integrated into larger, multi-purpose libraries ([s3prl](https://github.com/s3prl/s3prl), [SpeechBrain](https://github.com/speechbrain/speechbrain), etc.).

The aim of this repository is to develop an extremely lightweight (and understandable) test harness using PyTorch Lightning and Lhoste (speech-specific PyTorch dataloaders) to benchmark various optimizations on the [pure PyTorch (+Lightning) HuBERT pre-training recipe](https://github.com/pytorch/audio/tree/main/examples/self_supervised_learning) provided in torchaudio.

## Road map

### Stage 1: Supervised training

- [x] Add Lhotse's minimal working example with ESPNet to repository
- [ ] Modify ESPNet recipe for use with PyTorch lightning
  - [ ] Single GPU ok?
  - [ ] Multi-GPU ok?
  - [ ] AMP (fp16/bf16) ok?
- [ ] Implement HuBERT fine-tuning

### Stage 2: Self-supervised training

- [ ] Fold in torchaudio's HuBERT pre-training recipe into harness

### Stage 3: Optimize self-supervised training recipe

- [ ] Fold in proposals from MelHuBERT/DistilHuBERT/Crammed BERT/microBERT/etc.
