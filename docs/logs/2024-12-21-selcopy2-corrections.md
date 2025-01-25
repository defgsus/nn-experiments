---
tags: ["lm", "cnn"]
---

# Corrections of wrong *Very Selective Copying* experiments

Two corrections of experiment results in [Very Selective Copying](2024-12-15-selcopy2.md).


### Compare attention invention

[Original](2024-12-15-selcopy2.md#self-invented-queries-keys-and-values)

Reproduce with [convtext-qa-program-3ops-attn-invent.yml @ c0cf2763](https://github.com/defgsus/nn-experiments/blob/c0cf27632ade5a643b6fd3598c60e01c960b6138/experiments/textmask/qa-program/convtext-qa-program-3ops-attn-invent.yml)
```shell
git checkout c0cf2763
python exp.py experiments/textmask/qa-program/convtext-qa-program-3ops-attn-invent.yml run
python exp.py experiments/textmask/qa-program/convtext-qa-program-3ops-attn-invent.yml results -ec dil matrix_slug matrix_id bs opt lr norm meter gc act posemb nitem len nops vnops l ch ks -sc "validation_sample_error%-" -ic "train_sample_error%" "validation_sample_error%" -ac trial -acs "train_sample_error%" "validation_sample_error%"
```

This time made **10 runs** per setting!

| attn  | qkv   | attnact |   validation loss |   train_sample_error% |     (min) |   (max) |    (std) |   validation_sample_error% |     (min) |   (max) |    (std) |   model params |   train time (minutes) |   throughput |
|:------|:------|:--------|------------------:|----------------------:|----------:|--------:|---------:|---------------------------:|----------:|--------:|---------:|---------------:|-----------------------:|-------------:|
| 0,0,T | KV    | elu+1   |          0.279915 |                64.541 |   60.7422 | 67.8711 |   2.1683 |                    99.2775 |    98.965 | 99.5024 | 0.180377 |        229,632 |                   1.54 |     10,843/s |
| 0,0,T | QV    | elu+1   |          0.211112 |               39.7949 |   27.5391 | 63.1836 |  11.0987 |                     94.992 |   88.6445 | 99.4228 |  3.70722 |        229,632 |                   1.56 |     10,709/s |
| 0,0,1 | QK    |         |          0.269297 |               55.8984 |   8.20312 |   93.75 |   28.063 |                     89.365 |   24.5721 | 99.9602 |  23.9533 |        246,272 |                   1.79 |      9,301/s |
| 0,0,1 | QKV   |         |           0.14078 |                30.918 |     3.125 | 64.3555 |  20.2595 |                    68.4385 |     10.41 | 99.7213 |  28.0022 |        299,584 |                   1.83 |      9,090/s |
| 0,0,4 | QK    |         |         0.0877218 |               16.5918 |   1.26953 | 36.9141 |  12.0085 |                    47.7339 |   3.86146 | 99.2038 |  33.1776 |        246,272 |                   1.81 |      9,217/s |
| 0,0,8 | QK    |         |         0.0309062 |               6.21094 |  0.195312 | 30.6641 |  10.0744 |                    18.4873 |  0.965366 | 86.1166 |  28.9473 |        246,272 |                   1.83 |      9,104/s |
| 0,0,8 | QV    |         |         0.0225483 |               5.41992 |         0 | 13.2812 |  3.64532 |                    16.1644 |  0.248806 | 36.2062 |  9.91068 |        246,272 |                   1.84 |      9,048/s |
| 0,0,4 | QKV   |         |         0.0253542 |                6.2207 |  0.195312 | 36.4258 |  12.3809 |                    15.6947 | 0.0895701 | 82.9319 |  29.7359 |        299,584 |                   1.91 |      8,746/s |
| 0,0,4 | QV    |         |          0.013519 |               2.86133 |  0.390625 |    6.25 |  2.00878 |                    9.34415 |   1.16441 | 17.6951 |  6.16371 |        246,272 |                   1.85 |      9,045/s |
| 0,0,8 | KV    |         |         0.0103882 |               2.31445 |  0.292969 | 4.58984 |  1.36412 |                     7.2074 |   1.24403 | 13.6644 |  4.41778 |        246,272 |                   1.85 |      9,021/s |
| 0,0,1 | QV    |         |        0.00974547 |                2.2168 | 0.0976562 | 7.51953 |  2.48893 |                    6.87301 | 0.0796178 | 21.9845 |  7.43911 |        246,272 |                   1.77 |      9,414/s |
| 0,0,1 | KV    |         |        0.00557148 |               1.37695 |         0 | 4.10156 |  1.69861 |                    4.50239 |  0.139331 | 15.9236 |  5.87746 |        246,272 |                   1.77 |      9,413/s |
| 0,0,4 | KV    |         |        0.00294197 |                 0.625 | 0.0976562 | 1.75781 | 0.500498 |                    1.83917 |  0.338376 | 4.90645 |  1.34889 |        246,272 |                   1.86 |      8,985/s |
| 0,0,8 | QKV   |         |        0.00278822 |              0.527344 |         0 | 2.73438 | 0.804241 |                    1.58041 |  0.348328 | 6.49881 |  1.85126 |        299,584 |                    1.9 |      8,786/s |
| 0,0,T | QKV   | elu+1   |        0.00210139 |              0.380859 |         0 | 1.07422 | 0.299939 |                     1.2291 |  0.348328 |  2.2293 | 0.648963 |        282,944 |                    1.6 |     10,441/s |
| 0,0,T | QK    | elu+1   |          0.001886 |              0.488281 |         0 | 1.26953 | 0.387903 |                   0.936505 |  0.159236 | 1.98049 | 0.508104 |        229,632 |                   1.58 |     10,595/s |

- self-built attention 
  - `QK` or `QKV` perform most stable
  - `QV` and `KV` perform not at all
- multi-head attention
  - 8 heads `QKV` performs most stable


### Compare attention activation function

[Original](2024-12-15-selcopy2.md#compare-attention-activation-function)

Reproduce with [convtext-qa-program-3ops-attn-act.yml @ 6bedc896](https://github.com/defgsus/nn-experiments/blob/6bedc89667062f7f8d18d77df0b6bec253836bc8/experiments/textmask/qa-program/convtext-qa-program-3ops-attn-act.yml)

Here's the 5-runs average for each activation function.

| qkv   | attn  | attnact   |   validation loss |   validation_mask_error% |   validation_sample_error% |    (min) |   (max) |    (std) |   model params |   train time (minutes) |   throughput |
|:------|:------|:----------|------------------:|-------------------------:|---------------------------:|---------:|--------:|---------:|---------------:|-----------------------:|-------------:|
| QK    | 0,0,T | elu+1     |        0.00131346 |                 0.142118 |                   0.676752 | 0.199045 | 1.05494 | 0.346975 |        229,632 |                   1.51 |     11,066/s |
| QK    | 0,0,T | dpfp      |       0.000896075 |                 0.119029 |                   0.597134 |  0.17914 | 1.23408 |  0.54615 |        229,632 |                   1.69 |      9,871/s |


The **dpfp** seems to perform a little better (it's certainly converging a bit earlier in training)
but looking at the individual runs i retract and rather think there is nothing that can be said
with any evidence:

|   trial | qkv   | attn  | attnact |   validation loss | validation_mask_error% |   validation_sample_error% |   model params |   train time (minutes) |   throughput |
|--------:|:------|:------|:--------|------------------:|-----------------------:|---------------------------:|---------------:|-----------------------:|-------------:|
|       2 | QK    | 0,0,T | dpfp    |        0.00161164 |               0.248806 |                    1.23408 |        229,632 |                   1.67 |      9,979/s |
|       5 | QK    | 0,0,T | dpfp    |        0.00167819 |               0.230892 |                    1.15446 |        229,632 |                   1.76 |      9,491/s |
|       5 | QK    | 0,0,T | elu+1   |        0.00170212 |               0.222930 |                    1.05494 |        229,632 |                   1.61 |     10,347/s |
|       2 | QK    | 0,0,T | elu+1   |        0.00170096 |               0.191083 |                    0.88574 |        229,632 |                   1.47 |     11,317/s |
|       1 | QK    | 0,0,T | elu+1   |        0.00159251 |               0.163217 |                    0.79617 |        229,632 |                   1.33 |     12,507/s |
|       3 | QK    | 0,0,T | elu+1   |        0.00101384 |               0.095541 |                    0.44785 |        229,632 |                   1.54 |     10,842/s |
|       1 | QK    | 0,0,T | dpfp    |        0.00042169 |               0.045780 |                    0.22890 |        229,632 |                   1.59 |     10,498/s |
|       4 | QK    | 0,0,T | elu+1   |        0.00055788 |               0.037818 |                    0.19904 |        229,632 |                   1.62 |     10,314/s |
|       4 | QK    | 0,0,T | dpfp    |        0.00034492 |               0.035828 |                    0.18909 |        229,632 |                   1.74 |      9,552/s |
|       3 | QK    | 0,0,T | dpfp    |        0.00042392 |               0.033837 |                    0.17914 |        229,632 |                   1.69 |      9,834/s |
