# Reservoir computing

## Reproducing "Handwritten Digit Recognition by Spin Waves in a Skyrmion Reservoir"

by Mu-Kun Lee, Masahito Mochizuki [arxiv:2309.06815](https://arxiv.org/abs/2309.06815)

It's an interesting paper. I try to reproduce the digital part of it:
classifying the MNIST dataset via linear readout of an "Echo State Network".

Here are some generic tests for the "echo state". The used `Reservoir` model
is defined in [src/models/reservoir/reservoir.py](../../src/models/reservoir/reservoir.py).

The MNIST dataset is cropped and rearranged like in the paper with: 660 train and 330 test
images for each class, image size 22x16. 

The reservoir has 16 inputs which receive 22 image rows in consecutive order. 
The resulting state map of 22 x `num_cells` is fed to 
a `sklearn.linear_model.Ridge` with default parameters and fitted to the
image's class logits (10 numbers, all zero except one).

### Baseline without reservoir

Feeding the raw 22x16 images to the `Ridge` gives

| dataset | error l1 | accuracy |
|:--------|---------:|---------:|
| train   |    0.136 |    0.880 |
| test    |    0.149 |    0.824 |

### `Reservoir(num_inputs=16, num_cells=100)`

Each setup was run 5 times and the average is reported. Table is sorted by `test_accuracy`. 
Full runtime was 5min.

| id                                                               |   train_error_l1 |   train_accuracy |   test_error_l1 |   test_accuracy |
|:-----------------------------------------------------------------|-----------------:|-----------------:|----------------:|----------------:|
| activation=tanh rec_prob=0.1 rec_std=1.5 leak_rate=0.1           |        0.0913619 |         0.973182 |        0.11267  |        0.916545 |
| activation=tanh rec_prob=0.1 rec_std=1.0 leak_rate=0.1           |        0.0909993 |         0.972818 |        0.111962 |        0.916242 |
| activation=tanh rec_prob=0.1 rec_std=2.0 leak_rate=0.1           |        0.0904363 |         0.973939 |        0.11214  |        0.915212 |
| activation=tanh rec_prob=0.5 rec_std=0.5 leak_rate=0.1           |        0.0909891 |         0.972758 |        0.11204  |        0.91503  |
| activation=tanh rec_prob=0.5 rec_std=1.0 leak_rate=0.1           |        0.0910437 |         0.973909 |        0.113129 |        0.914909 |
| activation=tanh rec_prob=1.0 rec_std=0.5 leak_rate=0.1           |        0.0908132 |         0.974606 |        0.112472 |        0.914545 |
| activation=tanh rec_prob=0.1 rec_std=0.5 leak_rate=0.5           |        0.0814378 |         0.991667 |        0.120942 |        0.914303 |
| activation=sigmoid rec_prob=0.5 rec_std=1.0 leak_rate=(0.1, 0.9) |        0.0905591 |         0.974    |        0.112425 |        0.91303  |
| activation=tanh rec_prob=0.5 rec_std=1.5 leak_rate=0.1           |        0.0913657 |         0.973849 |        0.113769 |        0.912182 |
| activation=sigmoid rec_prob=0.5 rec_std=1.0 leak_rate=0.5        |        0.0919869 |         0.971667 |        0.114358 |        0.912121 |
| activation=sigmoid rec_prob=1.0 rec_std=1.0 leak_rate=(0.1, 0.9) |        0.0924256 |         0.972212 |        0.114403 |        0.911091 |
| activation=sigmoid rec_prob=0.1 rec_std=2.0 leak_rate=(0.1, 0.9) |        0.0946606 |         0.968242 |        0.115533 |        0.910667 |
| activation=sigmoid rec_prob=1.0 rec_std=1.5 leak_rate=(0.1, 0.9) |        0.0919614 |         0.973818 |        0.115689 |        0.91     |
| activation=sigmoid rec_prob=1.0 rec_std=1.0 leak_rate=0.5        |        0.0912574 |         0.97403  |        0.114051 |        0.91     |
| activation=sigmoid rec_prob=1.0 rec_std=1.5 leak_rate=0.5        |        0.0915149 |         0.975667 |        0.115996 |        0.909939 |
| activation=tanh rec_prob=1.0 rec_std=1.5 leak_rate=0.1           |        0.0927401 |         0.972394 |        0.115492 |        0.909879 |
| activation=sigmoid rec_prob=0.5 rec_std=2.0 leak_rate=(0.1, 0.9) |        0.0922088 |         0.974364 |        0.11642  |        0.909758 |
| activation=sigmoid rec_prob=0.1 rec_std=2.0 leak_rate=0.5        |        0.0931417 |         0.970242 |        0.114565 |        0.909636 |
| activation=tanh rec_prob=1.0 rec_std=1.0 leak_rate=0.1           |        0.0932079 |         0.973879 |        0.115654 |        0.909515 |
| activation=sigmoid rec_prob=1.0 rec_std=0.5 leak_rate=0.5        |        0.0935655 |         0.968909 |        0.114576 |        0.909455 |
| activation=tanh rec_prob=0.1 rec_std=0.5 leak_rate=(0.1, 0.9)    |        0.0846361 |         0.989879 |        0.122181 |        0.909333 |
| activation=sigmoid rec_prob=1.0 rec_std=2.0 leak_rate=(0.1, 0.9) |        0.0927946 |         0.974849 |        0.118656 |        0.90897  |
| activation=sigmoid rec_prob=0.5 rec_std=1.5 leak_rate=(0.1, 0.9) |        0.0927865 |         0.972212 |        0.115058 |        0.90897  |
| activation=sigmoid rec_prob=1.0 rec_std=0.5 leak_rate=(0.1, 0.9) |        0.0941282 |         0.967879 |        0.115    |        0.908364 |
| activation=sigmoid rec_prob=0.5 rec_std=1.5 leak_rate=0.5        |        0.092112  |         0.973545 |        0.115005 |        0.908303 |
| activation=sigmoid rec_prob=0.1 rec_std=1.5 leak_rate=(0.1, 0.9) |        0.0947395 |         0.968333 |        0.115974 |        0.908182 |
| activation=sigmoid rec_prob=0.1 rec_std=1.5 leak_rate=0.5        |        0.0936448 |         0.969576 |        0.114822 |        0.908061 |
| activation=sigmoid rec_prob=0.5 rec_std=2.0 leak_rate=0.5        |        0.0908134 |         0.975818 |        0.115763 |        0.907939 |
| activation=tanh rec_prob=0.5 rec_std=2.0 leak_rate=0.1           |        0.092655  |         0.973727 |        0.115907 |        0.907879 |
| activation=tanh rec_prob=0.1 rec_std=1.0 leak_rate=(0.1, 0.9)    |        0.0845372 |         0.991212 |        0.125265 |        0.90697  |
| activation=tanh rec_prob=0.1 rec_std=0.5 leak_rate=0.1           |        0.0940639 |         0.970545 |        0.115203 |        0.90697  |
| activation=sigmoid rec_prob=0.5 rec_std=1.5 leak_rate=0.1        |        0.10626   |         0.94497  |        0.117461 |        0.906606 |
| activation=sigmoid rec_prob=1.0 rec_std=2.0 leak_rate=0.1        |        0.104187  |         0.948212 |        0.116084 |        0.906303 |
| activation=tanh rec_prob=0.5 rec_std=0.5 leak_rate=(0.1, 0.9)    |        0.0857813 |         0.990576 |        0.126475 |        0.906    |
| activation=tanh rec_prob=0.1 rec_std=1.0 leak_rate=0.5           |        0.0837917 |         0.991788 |        0.126254 |        0.905939 |
| activation=tanh rec_prob=1.0 rec_std=2.0 leak_rate=0.1           |        0.0939976 |         0.972545 |        0.118332 |        0.905818 |
| activation=sigmoid rec_prob=0.5 rec_std=2.0 leak_rate=0.1        |        0.105004  |         0.947333 |        0.116669 |        0.905576 |
| activation=sigmoid rec_prob=1.0 rec_std=1.5 leak_rate=0.1        |        0.105035  |         0.947303 |        0.116582 |        0.905394 |
| activation=sigmoid rec_prob=1.0 rec_std=2.0 leak_rate=0.5        |        0.0938446 |         0.973758 |        0.11896  |        0.904667 |
| activation=sigmoid rec_prob=1.0 rec_std=1.0 leak_rate=0.1        |        0.106686  |         0.944121 |        0.118094 |        0.903818 |
| activation=sigmoid rec_prob=0.5 rec_std=0.5 leak_rate=0.5        |        0.0954133 |         0.966333 |        0.116633 |        0.903758 |
| activation=sigmoid rec_prob=0.5 rec_std=0.5 leak_rate=(0.1, 0.9) |        0.0966389 |         0.965939 |        0.117036 |        0.902667 |
| activation=sigmoid rec_prob=0.5 rec_std=1.0 leak_rate=0.1        |        0.108358  |         0.942576 |        0.119501 |        0.902    |
| activation=sigmoid rec_prob=0.1 rec_std=1.0 leak_rate=0.5        |        0.0967201 |         0.965636 |        0.117628 |        0.901697 |
| activation=sigmoid rec_prob=0.5 rec_std=1.0 leak_rate=0.9        |        0.090546  |         0.979364 |        0.120988 |        0.901636 |
| activation=tanh rec_prob=0.5 rec_std=0.5 leak_rate=0.5           |        0.0849166 |         0.992273 |        0.128532 |        0.901394 |
| activation=sigmoid rec_prob=0.1 rec_std=2.0 leak_rate=0.9        |        0.0912182 |         0.978303 |        0.120617 |        0.901273 |
| activation=sigmoid rec_prob=0.5 rec_std=1.5 leak_rate=0.9        |        0.0898494 |         0.981849 |        0.1218   |        0.901273 |
| activation=tanh rec_prob=0.1 rec_std=1.5 leak_rate=0.5           |        0.0859504 |         0.991727 |        0.130171 |        0.900303 |
| activation=sigmoid rec_prob=0.1 rec_std=2.0 leak_rate=0.1        |        0.107953  |         0.943212 |        0.11899  |        0.900121 |
| activation=sigmoid rec_prob=1.0 rec_std=1.0 leak_rate=0.9        |        0.090948  |         0.980667 |        0.122263 |        0.899576 |
| activation=sigmoid rec_prob=0.1 rec_std=1.0 leak_rate=(0.1, 0.9) |        0.097375  |         0.964212 |        0.1184   |        0.899515 |
| activation=sigmoid rec_prob=0.1 rec_std=1.5 leak_rate=0.9        |        0.0928208 |         0.974939 |        0.120598 |        0.898727 |
| activation=sigmoid rec_prob=0.1 rec_std=1.5 leak_rate=0.1        |        0.108945  |         0.941    |        0.120246 |        0.898545 |
| activation=sigmoid rec_prob=1.0 rec_std=0.5 leak_rate=0.9        |        0.0912399 |         0.977788 |        0.119822 |        0.898485 |
| activation=sigmoid rec_prob=1.0 rec_std=0.5 leak_rate=0.1        |        0.109638  |         0.940121 |        0.120644 |        0.898303 |
| activation=tanh rec_prob=0.1 rec_std=1.5 leak_rate=(0.1, 0.9)    |        0.0862176 |         0.990606 |        0.128655 |        0.898    |
| activation=tanh rec_prob=1.0 rec_std=0.5 leak_rate=(0.1, 0.9)    |        0.0875074 |         0.98997  |        0.131127 |        0.897758 |
| activation=sigmoid rec_prob=0.1 rec_std=1.0 leak_rate=0.1        |        0.109694  |         0.940364 |        0.120842 |        0.897455 |
| activation=sigmoid rec_prob=1.0 rec_std=1.5 leak_rate=0.9        |        0.0906575 |         0.982242 |        0.124433 |        0.897333 |
| activation=sigmoid rec_prob=0.5 rec_std=0.5 leak_rate=0.1        |        0.109821  |         0.940667 |        0.12124  |        0.897333 |
| activation=sigmoid rec_prob=0.5 rec_std=2.0 leak_rate=0.9        |        0.0918953 |         0.981636 |        0.12488  |        0.896788 |
| activation=sigmoid rec_prob=0.1 rec_std=1.0 leak_rate=0.9        |        0.0939872 |         0.972364 |        0.121576 |        0.896424 |
| activation=tanh rec_prob=0.1 rec_std=2.0 leak_rate=0.5           |        0.0875117 |         0.990727 |        0.132388 |        0.896303 |
| activation=tanh rec_prob=0.1 rec_std=0.5 leak_rate=0.9           |        0.0848733 |         0.993061 |        0.133139 |        0.895758 |
| activation=sigmoid rec_prob=0.1 rec_std=0.5 leak_rate=0.1        |        0.111193  |         0.938879 |        0.122395 |        0.895515 |
| activation=sigmoid rec_prob=0.5 rec_std=0.5 leak_rate=0.9        |        0.0938623 |         0.973394 |        0.121762 |        0.895455 |
| activation=tanh rec_prob=0.1 rec_std=2.0 leak_rate=(0.1, 0.9)    |        0.0881148 |         0.990091 |        0.131537 |        0.895455 |
| activation=tanh rec_prob=0.5 rec_std=1.0 leak_rate=(0.1, 0.9)    |        0.0890663 |         0.989909 |        0.132775 |        0.894182 |
| activation=tanh rec_prob=1.0 rec_std=0.5 leak_rate=0.5           |        0.0876476 |         0.99203  |        0.133196 |        0.893333 |
| activation=sigmoid rec_prob=0.1 rec_std=0.5 leak_rate=0.5        |        0.0990152 |         0.96297  |        0.120357 |        0.893333 |
| activation=tanh rec_prob=1.0 rec_std=1.0 leak_rate=(0.1, 0.9)    |        0.0910776 |         0.989242 |        0.13638  |        0.892909 |
| activation=sigmoid rec_prob=0.1 rec_std=0.5 leak_rate=(0.1, 0.9) |        0.100388  |         0.960121 |        0.121514 |        0.892121 |
| activation=sigmoid rec_prob=1.0 rec_std=2.0 leak_rate=0.9        |        0.0928298 |         0.980818 |        0.125962 |        0.892    |
| activation=tanh rec_prob=0.1 rec_std=1.0 leak_rate=0.9           |        0.0874595 |         0.992848 |        0.137025 |        0.890485 |
| activation=sigmoid rec_prob=0.1 rec_std=0.5 leak_rate=0.9        |        0.0955993 |         0.971727 |        0.123394 |        0.890303 |
| activation=tanh rec_prob=0.5 rec_std=1.5 leak_rate=(0.1, 0.9)    |        0.0913394 |         0.989364 |        0.136211 |        0.887758 |
| activation=tanh rec_prob=0.5 rec_std=2.0 leak_rate=0.5           |        0.0913921 |         0.990273 |        0.138816 |        0.886061 |
| activation=tanh rec_prob=0.5 rec_std=1.5 leak_rate=0.5           |        0.0911414 |         0.990061 |        0.138808 |        0.885515 |
| activation=tanh rec_prob=0.5 rec_std=0.5 leak_rate=0.9           |        0.0894512 |         0.991909 |        0.140217 |        0.885515 |
| activation=tanh rec_prob=0.5 rec_std=1.0 leak_rate=0.5           |        0.0899039 |         0.990394 |        0.136814 |        0.885333 |
| activation=tanh rec_prob=1.0 rec_std=1.0 leak_rate=0.5           |        0.0914824 |         0.990152 |        0.138184 |        0.884909 |
| activation=tanh rec_prob=1.0 rec_std=2.0 leak_rate=(0.1, 0.9)    |        0.0935883 |         0.988333 |        0.140191 |        0.884061 |
| activation=tanh rec_prob=0.5 rec_std=2.0 leak_rate=(0.1, 0.9)    |        0.0917636 |         0.989333 |        0.137269 |        0.883091 |
| activation=tanh rec_prob=0.1 rec_std=1.5 leak_rate=0.9           |        0.0908111 |         0.992121 |        0.141601 |        0.882242 |
| activation=tanh rec_prob=1.0 rec_std=1.5 leak_rate=(0.1, 0.9)    |        0.0937087 |         0.988091 |        0.140329 |        0.881818 |
| activation=tanh rec_prob=1.0 rec_std=1.5 leak_rate=0.5           |        0.0918961 |         0.98903  |        0.139636 |        0.879333 |
| activation=tanh rec_prob=1.0 rec_std=2.0 leak_rate=0.5           |        0.0934811 |         0.990758 |        0.141498 |        0.879273 |
| activation=tanh rec_prob=0.1 rec_std=2.0 leak_rate=0.9           |        0.0933614 |         0.99097  |        0.145027 |        0.876606 |
| activation=tanh rec_prob=1.0 rec_std=0.5 leak_rate=0.9           |        0.0934049 |         0.991212 |        0.144974 |        0.876545 |
| activation=tanh rec_prob=0.5 rec_std=1.0 leak_rate=0.9           |        0.0950379 |         0.990455 |        0.147789 |        0.868727 |
| activation=tanh rec_prob=1.0 rec_std=1.0 leak_rate=0.9           |        0.0965934 |         0.989727 |        0.149038 |        0.867333 |
| activation=tanh rec_prob=0.5 rec_std=1.5 leak_rate=0.9           |        0.0960845 |         0.989455 |        0.14909  |        0.865091 |
| activation=tanh rec_prob=0.5 rec_std=2.0 leak_rate=0.9           |        0.096801  |         0.990152 |        0.1499   |        0.864667 |
| activation=tanh rec_prob=1.0 rec_std=2.0 leak_rate=0.9           |        0.0985903 |         0.988909 |        0.151953 |        0.861394 |
| activation=tanh rec_prob=1.0 rec_std=1.5 leak_rate=0.9           |        0.098347  |         0.989182 |        0.151612 |        0.86103  |

### `Reservoir(num_inputs=16, num_cells=1000)`

Repeat everything with 1000 reservoir cells. Full runtime 67min.

| id                                                               |   train_error_l1 |   train_accuracy |   test_error_l1 |   test_accuracy |
|:-----------------------------------------------------------------|-----------------:|-----------------:|----------------:|----------------:|
| activation=sigmoid rec_prob=0.1 rec_std=1.0 leak_rate=0.5        |      0.0381668   |         0.999879 |       0.0982787 |        0.952424 |
| activation=sigmoid rec_prob=0.5 rec_std=0.5 leak_rate=0.5        |      0.037041    |         0.999879 |       0.0999229 |        0.949273 |
| activation=sigmoid rec_prob=0.1 rec_std=1.0 leak_rate=(0.1, 0.9) |      0.0387771   |         0.999848 |       0.0995846 |        0.948485 |
| activation=sigmoid rec_prob=0.1 rec_std=0.5 leak_rate=0.5        |      0.0475672   |         0.999485 |       0.0975823 |        0.947758 |
| activation=sigmoid rec_prob=0.5 rec_std=0.5 leak_rate=(0.1, 0.9) |      0.038247    |         0.999909 |       0.101271  |        0.947697 |
| activation=sigmoid rec_prob=0.1 rec_std=0.5 leak_rate=(0.1, 0.9) |      0.0473766   |         0.999667 |       0.0989504 |        0.946303 |
| activation=sigmoid rec_prob=0.1 rec_std=1.5 leak_rate=(0.1, 0.9) |      0.035789    |         0.99997  |       0.10656   |        0.945152 |
| activation=tanh rec_prob=0.1 rec_std=0.5 leak_rate=0.1           |      0.0366382   |         0.999939 |       0.103881  |        0.944667 |
| activation=sigmoid rec_prob=0.1 rec_std=1.5 leak_rate=0.5        |      0.0346061   |         0.999939 |       0.106813  |        0.944545 |
| activation=sigmoid rec_prob=1.0 rec_std=0.5 leak_rate=0.5        |      0.0337909   |         1        |       0.107468  |        0.943818 |
| activation=sigmoid rec_prob=1.0 rec_std=0.5 leak_rate=(0.1, 0.9) |      0.0342324   |         0.99997  |       0.108108  |        0.943333 |
| activation=sigmoid rec_prob=0.5 rec_std=1.0 leak_rate=0.1        |      0.068811    |         0.991818 |       0.0938989 |        0.94     |
| activation=sigmoid rec_prob=1.0 rec_std=1.0 leak_rate=0.1        |      0.0687783   |         0.992424 |       0.0948929 |        0.939636 |
| activation=sigmoid rec_prob=0.1 rec_std=2.0 leak_rate=0.5        |      0.0325047   |         1        |       0.114238  |        0.939455 |
| activation=sigmoid rec_prob=0.5 rec_std=1.5 leak_rate=0.1        |      0.068365    |         0.992091 |       0.0950397 |        0.938667 |
| activation=sigmoid rec_prob=0.1 rec_std=0.5 leak_rate=0.9        |      0.0355591   |         0.999879 |       0.110459  |        0.938667 |
| activation=sigmoid rec_prob=1.0 rec_std=1.5 leak_rate=0.1        |      0.0678975   |         0.993515 |       0.0964695 |        0.937515 |
| activation=sigmoid rec_prob=0.5 rec_std=2.0 leak_rate=0.1        |      0.067665    |         0.993424 |       0.0955492 |        0.937394 |
| activation=sigmoid rec_prob=0.5 rec_std=1.0 leak_rate=(0.1, 0.9) |      0.0320317   |         1        |       0.11756   |        0.936545 |
| activation=tanh rec_prob=0.1 rec_std=1.0 leak_rate=0.1           |      0.0331773   |         1        |       0.113007  |        0.936364 |
| activation=sigmoid rec_prob=0.1 rec_std=2.0 leak_rate=0.1        |      0.0695632   |         0.991212 |       0.0943548 |        0.936121 |
| activation=sigmoid rec_prob=0.1 rec_std=1.5 leak_rate=0.1        |      0.0711534   |         0.989394 |       0.0946351 |        0.935879 |
| activation=sigmoid rec_prob=1.0 rec_std=0.5 leak_rate=0.1        |      0.0705772   |         0.990121 |       0.0945419 |        0.935636 |
| activation=tanh rec_prob=0.5 rec_std=0.5 leak_rate=0.1           |      0.0328507   |         1        |       0.113785  |        0.935636 |
| activation=sigmoid rec_prob=0.1 rec_std=2.0 leak_rate=(0.1, 0.9) |      0.0334346   |         1        |       0.114923  |        0.935394 |
| activation=sigmoid rec_prob=1.0 rec_std=2.0 leak_rate=0.1        |      0.0676327   |         0.994485 |       0.0973725 |        0.935333 |
| activation=sigmoid rec_prob=0.5 rec_std=1.0 leak_rate=0.5        |      0.0327532   |         1        |       0.11662   |        0.934061 |
| activation=sigmoid rec_prob=0.5 rec_std=0.5 leak_rate=0.1        |      0.0733776   |         0.987424 |       0.0963458 |        0.933333 |
| activation=tanh rec_prob=0.1 rec_std=1.5 leak_rate=0.1           |      0.031549    |         1        |       0.11696   |        0.931879 |
| activation=sigmoid rec_prob=0.1 rec_std=1.0 leak_rate=0.9        |      0.0220503   |         1        |       0.121302  |        0.931879 |
| activation=sigmoid rec_prob=0.1 rec_std=1.0 leak_rate=0.1        |      0.074468    |         0.986848 |       0.0969672 |        0.931697 |
| activation=tanh rec_prob=1.0 rec_std=0.5 leak_rate=0.1           |      0.0318906   |         1        |       0.117717  |        0.93     |
| activation=tanh rec_prob=0.1 rec_std=0.5 leak_rate=0.5           |      0.00118011  |         1        |       0.131636  |        0.929697 |
| activation=tanh rec_prob=0.1 rec_std=0.5 leak_rate=(0.1, 0.9)    |      0.00137585  |         1        |       0.134411  |        0.92897  |
| activation=tanh rec_prob=1.0 rec_std=1.0 leak_rate=0.1           |      0.0289397   |         1        |       0.124778  |        0.927333 |
| activation=tanh rec_prob=0.5 rec_std=1.0 leak_rate=0.1           |      0.0304781   |         1        |       0.121512  |        0.927273 |
| activation=sigmoid rec_prob=1.0 rec_std=1.0 leak_rate=0.5        |      0.0301155   |         1        |       0.127572  |        0.927152 |
| activation=sigmoid rec_prob=0.5 rec_std=0.5 leak_rate=0.9        |      0.0203429   |         1        |       0.124832  |        0.926909 |
| activation=tanh rec_prob=0.1 rec_std=2.0 leak_rate=0.1           |      0.0306263   |         1        |       0.120241  |        0.926485 |
| activation=sigmoid rec_prob=0.5 rec_std=1.5 leak_rate=(0.1, 0.9) |      0.0310583   |         1        |       0.128808  |        0.924606 |
| activation=tanh rec_prob=0.5 rec_std=1.5 leak_rate=0.1           |      0.029312    |         1        |       0.125112  |        0.924242 |
| activation=sigmoid rec_prob=0.5 rec_std=1.5 leak_rate=0.5        |      0.0305194   |         1        |       0.128429  |        0.923939 |
| activation=sigmoid rec_prob=0.1 rec_std=0.5 leak_rate=0.1        |      0.0810556   |         0.982091 |       0.103256  |        0.923636 |
| activation=sigmoid rec_prob=1.0 rec_std=1.0 leak_rate=(0.1, 0.9) |      0.0303334   |         1        |       0.127283  |        0.923394 |
| activation=tanh rec_prob=0.1 rec_std=1.0 leak_rate=0.5           |      0.000661131 |         1        |       0.139365  |        0.921697 |
| activation=tanh rec_prob=1.0 rec_std=2.0 leak_rate=0.1           |      0.0270765   |         1        |       0.129459  |        0.920849 |
| activation=tanh rec_prob=0.5 rec_std=2.0 leak_rate=0.1           |      0.0278414   |         1        |       0.12763   |        0.920424 |
| activation=tanh rec_prob=1.0 rec_std=1.5 leak_rate=0.1           |      0.0282247   |         1        |       0.127867  |        0.92     |
| activation=tanh rec_prob=0.1 rec_std=1.5 leak_rate=0.5           |      0.000572814 |         1        |       0.14156   |        0.919212 |
| activation=tanh rec_prob=1.0 rec_std=0.5 leak_rate=0.5           |      0.00054803  |         1        |       0.142264  |        0.918667 |
| activation=sigmoid rec_prob=1.0 rec_std=1.5 leak_rate=(0.1, 0.9) |      0.0304561   |         1        |       0.136521  |        0.918485 |
| activation=tanh rec_prob=0.1 rec_std=0.5 leak_rate=0.9           |      0.000173226 |         1        |       0.135079  |        0.918424 |
| activation=sigmoid rec_prob=0.5 rec_std=2.0 leak_rate=0.5        |      0.0286785   |         1        |       0.136517  |        0.918    |
| activation=tanh rec_prob=0.5 rec_std=0.5 leak_rate=0.5           |      0.000610946 |         1        |       0.139934  |        0.917697 |
| activation=sigmoid rec_prob=0.5 rec_std=2.0 leak_rate=(0.1, 0.9) |      0.0296537   |         1        |       0.136146  |        0.915939 |
| activation=sigmoid rec_prob=1.0 rec_std=1.5 leak_rate=0.5        |      0.0283219   |         1        |       0.138181  |        0.915212 |
| activation=tanh rec_prob=1.0 rec_std=1.5 leak_rate=0.5           |      0.000447909 |         1        |       0.145044  |        0.914606 |
| activation=tanh rec_prob=0.5 rec_std=1.5 leak_rate=0.5           |      0.000464348 |         1        |       0.145163  |        0.914606 |
| activation=tanh rec_prob=0.5 rec_std=0.5 leak_rate=(0.1, 0.9)    |      0.000651396 |         1        |       0.145794  |        0.914485 |
| activation=sigmoid rec_prob=0.1 rec_std=1.5 leak_rate=0.9        |      0.0175031   |         1        |       0.138835  |        0.914364 |
| activation=tanh rec_prob=0.1 rec_std=1.0 leak_rate=(0.1, 0.9)    |      0.000837996 |         1        |       0.145143  |        0.914    |
| activation=tanh rec_prob=1.0 rec_std=1.0 leak_rate=0.5           |      0.000457013 |         1        |       0.145044  |        0.913091 |
| activation=tanh rec_prob=0.1 rec_std=2.0 leak_rate=0.5           |      0.000502271 |         1        |       0.143849  |        0.912849 |
| activation=tanh rec_prob=0.5 rec_std=1.0 leak_rate=0.5           |      0.000475301 |         1        |       0.144441  |        0.912788 |
| activation=tanh rec_prob=0.5 rec_std=2.0 leak_rate=0.5           |      0.000444623 |         1        |       0.145215  |        0.912606 |
| activation=sigmoid rec_prob=1.0 rec_std=0.5 leak_rate=0.9        |      0.0147044   |         1        |       0.143055  |        0.911879 |
| activation=tanh rec_prob=0.1 rec_std=1.5 leak_rate=(0.1, 0.9)    |      0.000675669 |         1        |       0.148757  |        0.911394 |
| activation=tanh rec_prob=0.5 rec_std=1.0 leak_rate=(0.1, 0.9)    |      0.000550349 |         1        |       0.150015  |        0.910545 |
| activation=tanh rec_prob=1.0 rec_std=0.5 leak_rate=(0.1, 0.9)    |      0.000584363 |         1        |       0.148396  |        0.910424 |
| activation=tanh rec_prob=1.0 rec_std=2.0 leak_rate=0.5           |      0.000457479 |         1        |       0.146638  |        0.909818 |
| activation=tanh rec_prob=1.0 rec_std=1.0 leak_rate=(0.1, 0.9)    |      0.000553626 |         1        |       0.151664  |        0.907515 |
| activation=tanh rec_prob=0.1 rec_std=1.0 leak_rate=0.9           |      0.000110608 |         1        |       0.143074  |        0.906848 |
| activation=sigmoid rec_prob=1.0 rec_std=2.0 leak_rate=0.5        |      0.0258364   |         1        |       0.14669   |        0.906667 |
| activation=tanh rec_prob=0.1 rec_std=2.0 leak_rate=(0.1, 0.9)    |      0.000650342 |         1        |       0.150516  |        0.906606 |
| activation=tanh rec_prob=0.5 rec_std=1.5 leak_rate=(0.1, 0.9)    |      0.000544296 |         1        |       0.152909  |        0.906545 |
| activation=sigmoid rec_prob=1.0 rec_std=2.0 leak_rate=(0.1, 0.9) |      0.0297879   |         1        |       0.142981  |        0.906061 |
| activation=tanh rec_prob=0.5 rec_std=2.0 leak_rate=(0.1, 0.9)    |      0.00050405  |         1        |       0.153376  |        0.903939 |
| activation=sigmoid rec_prob=0.1 rec_std=2.0 leak_rate=0.9        |      0.0151607   |         1        |       0.151264  |        0.902788 |
| activation=tanh rec_prob=0.5 rec_std=0.5 leak_rate=0.9           |      0.00010102  |         1        |       0.144658  |        0.901636 |
| activation=tanh rec_prob=1.0 rec_std=2.0 leak_rate=(0.1, 0.9)    |      0.000494046 |         1        |       0.154521  |        0.901515 |
| activation=tanh rec_prob=0.1 rec_std=1.5 leak_rate=0.9           |      9.95549e-05 |         1        |       0.144931  |        0.900606 |
| activation=tanh rec_prob=1.0 rec_std=1.5 leak_rate=(0.1, 0.9)    |      0.000508447 |         1        |       0.1539    |        0.900061 |
| activation=tanh rec_prob=1.0 rec_std=0.5 leak_rate=0.9           |      9.21872e-05 |         1        |       0.146038  |        0.895636 |
| activation=tanh rec_prob=0.1 rec_std=2.0 leak_rate=0.9           |      8.88148e-05 |         1        |       0.146944  |        0.894424 |
| activation=tanh rec_prob=0.5 rec_std=1.0 leak_rate=0.9           |      8.50776e-05 |         1        |       0.147753  |        0.893818 |
| activation=tanh rec_prob=1.0 rec_std=1.0 leak_rate=0.9           |      8.35833e-05 |         1        |       0.148437  |        0.892909 |
| activation=tanh rec_prob=1.0 rec_std=2.0 leak_rate=0.9           |      8.17226e-05 |         1        |       0.14951   |        0.891879 |
| activation=tanh rec_prob=0.5 rec_std=1.5 leak_rate=0.9           |      8.39818e-05 |         1        |       0.148282  |        0.891515 |
| activation=sigmoid rec_prob=0.5 rec_std=1.0 leak_rate=0.9        |      0.0140698   |         1        |       0.15804   |        0.891515 |
| activation=tanh rec_prob=1.0 rec_std=1.5 leak_rate=0.9           |      8.32554e-05 |         1        |       0.148759  |        0.890364 |
| activation=tanh rec_prob=0.5 rec_std=2.0 leak_rate=0.9           |      8.46541e-05 |         1        |       0.150103  |        0.887697 |
| activation=sigmoid rec_prob=1.0 rec_std=1.0 leak_rate=0.9        |      0.011545    |         1        |       0.175481  |        0.869879 |
| activation=sigmoid rec_prob=0.5 rec_std=1.5 leak_rate=0.9        |      0.012171    |         1        |       0.177401  |        0.864    |
| activation=sigmoid rec_prob=0.5 rec_std=2.0 leak_rate=0.9        |      0.0107752   |         1        |       0.185805  |        0.852849 |
| activation=sigmoid rec_prob=1.0 rec_std=1.5 leak_rate=0.9        |      0.0106771   |         1        |       0.186955  |        0.848667 |
| activation=sigmoid rec_prob=1.0 rec_std=2.0 leak_rate=0.9        |      0.00964606  |         1        |       0.194721  |        0.835576 |

