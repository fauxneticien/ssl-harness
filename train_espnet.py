"""## Prepare data manifests"""

from lhotse import CutSet
from lhotse.dataset.collation import TokenCollater
from lhotse.recipes import prepare_librispeech

# Assuming 'download_mini-librispeech.py' has been run
libri = prepare_librispeech(corpus_dir='data/mini-librispeech/LibriSpeech')

cuts_train = CutSet.from_manifests(**libri['train-clean-5'])
cuts_dev = CutSet.from_manifests(**libri['dev-clean-2'])

"""## Setting up the "tokenizer" (simply use characters)"""

tokenizer = TokenCollater(cuts_train)

"""## Setting up a basic ESPnet Conformer"""

import argparse
from espnet.bin.asr_train import get_parser
from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E

parser = get_parser()
parser = E2E.add_arguments(parser)
config = parser.parse_args([
    "--mtlalpha", "0.0",  # weight for cross entropy and CTC loss
    "--outdir", "out", "--dict", ""])  # TODO: allow no arg

idim = 80
odim = len(tokenizer.idx2token)
setattr(config, "char_list", list(tokenizer.idx2token))
model = E2E(idim, odim, config)

"""## Define a minimal ESPnet Dataset class that uses Lhotse
"""

import torch
from lhotse import Fbank, FbankConfig
from lhotse.dataset import OnTheFlyFeatures

class MinimalEspnetDataset(torch.utils.data.Dataset):
  def __init__(self, tokenizer: TokenCollater):
    self.extractor = OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80)))
    self.tokenizer = tokenizer
  def __getitem__(self, cuts: CutSet) -> dict:
    cuts = cuts.sort_by_duration()
    feats, feat_lens = self.extractor(cuts)
    tokens, token_lens = self.tokenizer(cuts)
    return {
        'xs_pad': feats,
        'ilens': feat_lens,
        'ys_pad': tokens
    }

"""## Set up the data pipeline: sampler + dataset + dataloader"""

from lhotse.dataset import BucketingSampler
# We use BucketingSampler which will help minimize the padding.
train_sampler = BucketingSampler(cuts_train, max_duration=300, shuffle=True)
dev_sampler = BucketingSampler(cuts_dev, max_duration=400, shuffle=False)

dset = MinimalEspnetDataset(tokenizer)

train_dloader = torch.utils.data.DataLoader(dset, sampler=train_sampler, batch_size=None, num_workers=2)
dev_dloader = torch.utils.data.DataLoader(dset, sampler=dev_sampler, batch_size=None, num_workers=2)

"""## Training loop (minimal example)"""

import numpy as np

optim = torch.optim.Adam(model.parameters())
model.cuda()

train_acc = []
valid_acc = []

for epoch in range(10):
  # training
    acc = []
    model.train()

    train_dloader.sampler.set_epoch(epoch)

    for batch_idx, data in enumerate(train_dloader):
        loss = model(*[d.cuda() for d in data.values()])

        if batch_idx % 10 == 0:
           print(f'Batch {batch_idx} => loss {loss}')

        optim.zero_grad()
        loss.backward()
        acc.append(model.acc)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optim.step()
        t_r = loss.item()

    train_acc.append(np.mean(acc))
    # validation
    acc = []
    model.eval()
    for data in dev_dloader:
        model(*[d.cuda() for d in data.values()])
        acc.append(model.acc)
    valid_acc.append(np.mean(acc))
    print(f"epoch: {epoch}, train acc: {train_acc[-1]:.3f}, dev acc: {valid_acc[-1]:.3f}, loss:{t_r:.3f}")
