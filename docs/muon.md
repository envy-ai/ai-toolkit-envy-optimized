# Muon optimizer

Select **Muon** in the existing optimizer menu, or set `train.optimizer: muon` in a job configuration. Existing jobs and new-job defaults keep their selected optimizer. The option uses `toolkit.optimizers.muon.BatchedMuon` on the installed PyTorch 2.7 environment and needs no additional package.

Muon maintains one momentum buffer per trained parameter. Each matrix update is orthogonalized with five Newton–Schulz iterations by default. Matrices of the same flattened shape, dtype, and device are processed together; each matrix has its own norm. Batches are capped at 64 items and about 16 MiB of stacked bfloat16 input. Convolution tensors are flattened to `(output_channels, remaining_dimensions)` for the update and restored to their original shape. Vectors and scalars use the reference Muon's RMS-normalized momentum rule. Sparse gradients raise an error.

`train.optimizer_params` accepts `weight_decay`, `momentum`, `nesterov`, `ns_coefficients`, `eps`, `ns_steps`, and `adjust_lr_fn`. Default values are `0.01`, `0.95`, `true`, `(3.4445, -4.7750, 2.0315)`, `1e-7`, `5`, and `match_rms_adamw`, respectively. The default learning-rate adjustment matches the reference Fizgig optimizer wrapper, so existing AdamW learning rates are a reasonable starting point. `original` (or `null`) and `spectral_unclamped` are also available through job configuration. The UI's Weight Decay field continues to supply its existing value, `1e-4` for a new job, when Muon is selected.

Muon uses the ordinary optimizer parameter groups and honors their individual learning rates. Its momentum buffers live in the normal PyTorch optimizer state, so the existing checkpoint, resume, scheduler, and optimizer-state offload paths apply. Resume should use the same optimizer and parameter layout as the saved checkpoint. Full model training can carry much larger momentum buffers than adapter training; the optimizer does not switch a full fine-tune to another algorithm.

This implementation follows Fizgig's `src/fizgig/training/muon.py`, which adapts PyTorch's `torch/optim/_muon.py` (BSD 3-Clause). The required PyTorch notice follows.

## PyTorch (`torch.optim.Muon`) — BSD 3-Clause License

Upstream: https://github.com/pytorch/pytorch

`src/fizgig/training/muon.py` is adapted from PyTorch's `torch/optim/_muon.py` (2.10), used when `torch.optim.Muon` itself can't take the parameters. PyTorch's licence:

```
From PyTorch:

Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

From Caffe2:

Copyright (c) 2016-present, Facebook Inc. All rights reserved.

All contributions by Facebook:
Copyright (c) 2016 Facebook Inc.

All contributions by Google:
Copyright (c) 2015 Google Inc.
All rights reserved.

All contributions by Yangqing Jia:
Copyright (c) 2015 Yangqing Jia
All rights reserved.

All contributions by Kakao Brain:
Copyright 2019-2020 Kakao Brain

All contributions by Cruise LLC:
Copyright (c) 2022 Cruise LLC.
All rights reserved.

All contributions by Tri Dao:
Copyright (c) 2024 Tri Dao.
All rights reserved.

All contributions by Arm:
Copyright (c) 2021, 2023-2025 Arm Limited and/or its affiliates

All contributions from Caffe:
Copyright(c) 2013, 2014, 2015, the respective contributors
All rights reserved.

All other contributions:
Copyright(c) 2015, 2016 the respective contributors
All rights reserved.

Caffe2 uses a copyright model similar to Caffe: each contributor holds
copyright over their contributions to Caffe2. The project versioning records
all such contribution and copyright details. If a contributor wants to further
mark their specific copyright on a particular contribution, they should
indicate their copyright solely in the commit message of the change when it is
committed.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
   and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```
