# Corrections of wrong *Very Selective Copying* experiments


### Compare attention activation function

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


### Compare attention invention

Reproduce with [convtext-qa-program-3ops-attn-invent.yml @ 6bedc896](https://github.com/defgsus/nn-experiments/blob/6bedc89667062f7f8d18d77df0b6bec253836bc8/experiments/textmask/qa-program/convtext-qa-program-3ops-attn-invent.yml)

Still running...

