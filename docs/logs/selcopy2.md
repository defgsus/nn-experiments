# 2024-12-15 Solving the "Very Selective Copying" problem with a Very Small Language Model

To get a grip on the details, please check ["Selective Copying"](selcopy.md) first.

Basically, it's a synthetic question-answer dataset that requires some *"computational"* skill.
See, if you can find out the rule by yourself:

```
NA: 2>1: AN   
DQ: -1: Q    
EJWHB: 3>5: EJBHW
ULHP: 3>4, 2>1: LUPH 
YNP: 3>1, -3, 2>1: NP   
EJACQ: 1>2, -1, +3: EAJCQ
YESBR: 3>5, 5>1, 1>5, 3>4: YEBRS
UXMP: -2, -3, 1>2, +2: MPU  
```

**Spoiler**: First few letters are the program input, the numbers and operators are the program 
and answer, after the second colon, is the result of the program. 
`x>y` means: exchange `x`th item with `y`th item. `-x` means: remove `x`th item from memory and 
put it into the stack. `+x` means: pop last item from stack and insert at position `x`.

I argue, that this dataset
- requires a large enough *receptive field*
- requires some cognitional ability to build up a history and to use it subsequently. 

### Preliminary tests

Just to get a feeling for the dataset, i took the best model from the 
["Selective Copying"](selcopy.md) experiment with a few parameter variations 
and ran it on questions that have an input length of 2 to 5 items and 1 to 5 operations:

![error curves](img/selcopy2/selcopy2_mask-error_l6-12-18.png)

Training now takes 4 million steps, and could even be a bit longer. But generally, the loss curves
seem to converge at about there. 

|   nitem | len   | nops   |   l |   ch |   ks | dil                                 |  validation loss |   validation_mask_error% | model params | train time (minutes) |  throughput |
|--------:|:------|:-------|----:|-----:|-----:|:------------------------------------|-----------------:|-------------------------:|-------------:|---------------------:|------------:|
|      26 | 2,5   | 1,5    |   6 |   64 |    9 | 1,2,3,4,5,1                         |        0.0525558 |                  17.5876 |      237,952 |                15.30 |     4,358/s |
|      26 | 2,5   | 1,5    |   6 |  128 |    9 | 1,2,3,4,5,1                         |        0.0365154 |                  11.8352 |      918,272 |                10.76 |     6,195/s |
|      26 | 2,5   | 1,5    |  12 |   64 |    9 | 1,1,2,2,3,3,4,4,5,5,1,1             |        0.0076271 |                   2.0063 |      459,520 |                27.05 |     2,464/s |
|      26 | 2,5   | 1,5    |  18 |   64 |    9 | 1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,1,1,1 |        0.0028902 |                   0.7583 |      681,088 |                39.80 |     1,675/s |
|      26 | 2,5   | 1,5    |  12 |  128 |    9 | 1,1,2,2,3,3,4,4,5,5,1,1             |        0.0028549 |                   0.7205 |    1,803,776 |                18.81 |     3,543/s |

I'm using dilation all the time, because 1.) it's required for the receptive field of the 
6-layer network and 2.) because it runs faster. For the 12 and 18-layer networks i just expanded the
dilation settings without much thinking about it. (Actually i did think quite a bit
about it but decided to evaluate good 12 or 18-layer dilation settings another time)

Here are examples from the validation set from the worst and the best performing network:

#### 6-layer/32-chan: validation mask error 17.6%

![validation example output](img/selcopy2/selcopy2_validation-example_nops1-5_l6-ch64.png)

It is noticeable that long programs seem to be the problem for this network.

#### 12-layer/128-chan: validation mask error 0.7%

![validation example output](img/selcopy2/selcopy2_validation-example_nops1-5_l12-ch128.png)

Wow! Everything correct, even the long ones!

### Number of operations versus number of layers

The above example is a bit messy because it has a variable number of operations and 
we don't know how many answers are fully correct, only how many characters.
So first, add a new evaluation metric `validation_sample_error%`. While the mask error gives
the percentage of wrong characters within the mask area, the sample error gives the percentage
of wrong answers, even if only one character is wrong.

Further, to better evaluate the relationship between number of operations (length of the "program") 
and the number of layers in the network, i set a fixed number of operations.

I also dropped the stack operations (`+`/`-`) for now and made it all a bit tighter because 
i'm not interested in the influence of the receptive field in this experiment.
Example data with 5 operations looks like this:

    NPFKT:5>4 3>1 1>4 4>2 1>3:NFTPK
    UKLRM:5>1 4>2 1>4 2>4 3>2:KLMRU
    LCEWI:3>5 3>2 4>2 3>1 3>5:CWEIL
    LRUPX:1>4 3>2 5>4 1>3 2>1:URPXL
    UQJOR:5>4 2>5 5>4 4>3 2>5:URQJO
    BYGMJ:5>4 4>3 2>3 5>2 5>3:BMJGY
    PBNFM:3>5 5>1 1>3 2>3 5>3:MNPFB
    TNEOL:1>3 4>1 4>1 4>3 5>3:ENLTO
    ALEVT:1>2 2>1 2>1 4>2 4>3:LVAET
    UIZNQ:3>4 2>3 5>3 4>1 1>5:INQUZ

Running the networks on 2, 3, 4 and 5 operations per question:

#### 6-layer

![error curves](img/selcopy2/selcopy2_error-curves_l6-nops-2-5.png)

#### 12-layer

![error curves](img/selcopy2/selcopy2_error-curves_l12-nops-2-5.png)

#### 18-layer

![error curves](img/selcopy2/selcopy2_error-curves_l18-nops-2-5.png)

And here is it all together in a table. The dilation settings are left out for readability.
They are `1,2,3,4,5,1` for the 6-layer, `1,2,3,4,5,1,2,3,4,5,1,1` for the 12-layer
and `1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,1,1` for the 18-layer network.

|   nitem |   len |   nops |   l |   ch |   ks |   validation loss |   validation_mask_error% |   validation_sample_error% |   model params | train time (minutes) |  throughput |
|--------:|------:|-------:|----:|-----:|-----:|------------------:|-------------------------:|---------------------------:|---------------:|---------------------:|------------:|
|      26 |     5 |      5 |   6 |   64 |    9 |           0.23663 |               71.9705    |                  99.9104   |        237,952 |                58.04 |     1,145/s |
|      26 |     5 |      5 |  12 |   64 |    9 |          0.208893 |               63.6883    |                  99.4327   |        459,520 |                 40.0 |     1,660/s |
|      26 |     5 |      5 |  18 |   64 |    9 |          0.163338 |               50.2667    |                  98.547    |        681,088 |                 99.0 |       670/s |
|      26 |     5 |      4 |   6 |   64 |    9 |          0.185987 |               48.7739    |                  98.5967   |        237,952 |                 15.1 |     4,413/s |
|      26 |     5 |      4 |  12 |   64 |    9 |         0.0415355 |                8.8953    |                  37.8682   |        459,520 |                26.03 |     2,561/s |
|      26 |     5 |      3 |   6 |   64 |    9 |        0.00458625 |                0.559315  |                   2.75677  |        237,952 |                 7.32 |     9,109/s |
|      26 |     5 |      4 |  18 |   64 |    9 |       0.000329348 |                0.0378185 |                   0.189092 |        681,088 |                37.13 |     1,795/s |
|      26 |     5 |      3 |  18 |   64 |    9 |       1.86865e-06 |                0         |                   0        |        681,088 |                 5.07 |    13,139/s |
|      26 |     5 |      3 |  12 |   64 |    9 |        3.1804e-06 |                0         |                   0        |        459,520 |                 11.5 |     5,796/s |
|      26 |     5 |      2 |  18 |   64 |    9 |       4.39884e-07 |                0         |                   0        |        681,088 |                 15.3 |     4,357/s |
|      26 |     5 |      2 |   6 |   64 |    9 |        5.3105e-07 |                0         |                   0        |        237,952 |                 7.15 |     9,319/s |
|      26 |     5 |      2 |  12 |   64 |    9 |        1.0319e-06 |                0         |                   0        |        459,520 |                 11.3 |     5,900/s |

Something is obviously very strange. None of the networks can handle 5 operations. 
But some of them just did in the last experiment??

What i really like about this neural net research is that you need a lot of intuition to go ahead.
We simply don't know enough until all tests are made and this is usually hard to do because
it takes a lot of time. However, in this case, i would argue that a 5 operations sequence is
just too long for the network to learn the task in reasonable time. Put yourself into 
the eyes of the model. The training algorithm constantly tells you:

> If you see this: `PHMTW:5>3 2>3 5>4 5>3 1>3:?????`
> you should output that: `PHMTW:5>3 2>3 5>4 5>3 1>3:TWPMH`

on and on ... for millions of examples. Now, for a human, maybe a programmer even, this would
be solvable pretty quick. Once you have seen 10 examples, you can be pretty sure what the
algorithm is. But for a neural network trained with *stochastic gradient descend* it is basically
a *brute-force* approach. It's like: Hey network, you gotta lot of parameters, should be fine,
now if you see this X, and you should output that Y, there is a chance that inreasing this one 
parameter while decreasing that other one would help you to output the correct answer, the 
next time you see the same question. And on and on...

Once again, intuitively, the network can probably learn the task much easier when it has
some easier examples in the dataset. Schools and such also don't teach children algebra by 
starting with: *What is the answer of 7 + 4 + 9 + 2 + 6 = ??*

Now the training set contains questions with 2 to 5 operations, e.g.:

    HGOKX:3>1 5>4:OGHXK
    CXZMN:5>4 2>5:CMZNX
    RMYTC:2>1 4>5 4>2:MCYRT
    DNAHL:1>3 3>4 1>2 4>1 3>2:DHANL
    HSBQA:4>5 4>5:HSBQA
    DTFKS:4>5 3>2 4>5 5>1:SFTKD
    UXGJN:4>2 1>2 1>4 3>1 3>2:GXUJN

and the validation set still holds the same 5-operations-only questions for which all networks have
failed previously.

Here's the loss and error curves of the 18-layer network trained with 5 and 2-5 operations:

![error curves](img/selcopy2/selcopy2_error-curves_l18-nops-2-5-vnops5.png)

The bright one is the brighter one. Indeed, the 18-layer gets down to 2.5% sample error. It is
able to solve the task, it could just not learn it from the previous dataset. One could argue,
though, that it would eventually learn the task just from the 5-operations examples but it would
probably take 10 to 100 times more computation / CO2-emissions / floodings / draughts / you-name-it.

|   nitem |   len | nops   | val-nops |   l |   ch |   ks |   validation loss |   validation_mask_error% |   validation_sample_error% | model params   |   train time (minutes) | throughput   |
|--------:|------:|:-------|---------:|----:|-----:|-----:|------------------:|-------------------------:|---------------------------:|:---------------|-----------------------:|:-------------|
|      26 |     5 | 2,5    |        5 |   6 |   64 |    9 |          0.208661 |                60.0816   |                    99.5123 | 237,952        |                  16.55 | 4,027/s      |
|      26 |     5 | 2,5    |        5 |  12 |   64 |    9 |         0.0322987 |                 6.74761  |                    27.3885 | 459,520        |                  27.59 | 2,416/s      |
|      26 |     5 | 2,5    |        5 |  18 |   64 |    9 |        0.00329394 |                 0.565287 |                     2.5577 | 681,088        |                  41.25 | 1,615/s      |

We can see from the table, that the 12 and especially the 6-layer networks are struggling. 
Looking at the plots of the 6-layer networks trained with 5 and 2-5 operations, we can 
see that the mask error decreases by a good amount but actual sample error stays roughly
the same. It learned to put some letters and the right place but still fails
for almost every validation sample:

![error curves](img/selcopy2/selcopy2_error-curves_l6-nops-2-5-vnops5.png)

- Quick takeaway: **Put also some easy examples in the training set!**

The curves suggest, however, that training has not yet converged. There are sure a few more per-mille
to squeeze out.

This is a good point of origin for further experimentation. 
Can we get the 6-layer network to solve the *5-operations Very Selective Copying* problem,
- without adding so many modules that it actually resembles a 12-layer network
- without making it slower to execute than the 12-layer network
- ***bonus***: by keeping the number of model parameters equal or even lower

In other words, is there a trick, maybe to pass data around in a different way, 
that strongly increases the computational performance?

### Quck comparison with Mamba and LSTM

Just for another set of baselines, i tried the [state-spaces/Mamba](https://github.com/state-spaces/mamba)
(yet the [slow version](https://github.com/johnma2006/mamba-minimal))
and the all-beloved LSTM (as [pytorch implementation](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)). 

![error curves](img/selcopy2/selcopy2_error-curves_l6-nops-2-5-mamba-and-lstm.png)

The yellow/brownish is the 6-layers model from above, green is Mamba and blue the LSTM. 
All of them have 6 layers

| nitem |   len | nops   | val-nops |   l | model                           |  validation loss | validation_mask_error% |  validation_sample_error% |  model params |   train time (minutes) | throughput |
|------:|------:|:-------|---------:|----:|:--------------------------------|-----------------:|-----------------------:|--------------------------:|--------------:|-----------------------:|-----------:|
|    26 |     5 | 2,5    |        5 |   6 | LSTM hidden_size=64             |         0.264829 |                79.9602 |                       100 |       216,064 |                   6.72 |    9,926/s |
|    26 |     5 | 2,5    |        5 |   6 | Conv1d channels=64 (from above) |         0.208661 |                60.0816 |                   99.5123 |       237,952 |                  16.55 |    4,027/s |
|    26 |     5 | 2,5    |        5 |   6 | MAMBA d_model=32 d_state=16     |         0.199433 |                55.8081 |                   99.1839 |        67,936 |                 126.56 |      526/s |

- None of these especially-crafted models reached a significant performance gain. 
- To be fair: The Mamba model is *very* small compared to the Conv1d, just 28% of parameters. 
  Though it manages to consume 7.8x more computation time. The first tiny drop in sample error 
  occurred after 1 hour of training and i do not have the patience today to train an 
  equally-sized Mamba for comparison. (The dynamics of the curves suggest, that it would not change much)
- The LSTM basically archived nothing (though equal-sized). It's about as bad as the 6-layer 
  Conv1d with only 5-operations questions in the training set. At least, it's blazinlgy fast!
- Disclaimer: I do not know any best practices about using or training LSTMs or Mambas and
  surely made some mistake..


### Attention

![error curves](img/selcopy2/selcopy2_error-curves_l5-vnops5-attention.png)


|   nitem |   len | nops   |   vnops |   l |   ch |   ks | attn      |   validation loss | validation_mask_error% | validation_sample_error% | model params   |   train time (minutes) | throughput   |
|--------:|------:|:-------|--------:|----:|-----:|-----:|:----------|------------------:|-----------------------:|-------------------------:|:---------------|-----------------------:|:-------------|
|      26 |     5 | 2,5    |       5 |   5 |   64 |    9 | 0,0,4,0,0 |          0.010063 |                2.09793 |                  9.92237 | 291,520        |                  13.09 | 5,094/s      |
|      26 |     5 | 2,5    |       5 |   5 |   64 |    9 | 0,0,0,0,4 |        0.00499033 |               0.959395 |                  4.76712 | 291,520        |                    8.3 | 8,033/s      |
|      26 |     5 | 2,5    |       5 |   5 |   64 |    9 | 0,4,0,0,4 |       0.000208627 |              0.0278662 |                 0.139331 | 382,016        |                  11.36 | 5,867/s      |
|      26 |     5 | 2,5    |       5 |   5 |   64 |    9 | 0,4,4,0,0 |        0.00018748 |              0.0238854 |                 0.119427 | 382,016        |                  14.79 | 4,506/s      |
|      26 |     5 | 2,5    |       5 |   5 |   64 |    9 | 0,0,4,4,4 |       0.000129913 |              0.0199045 |                0.0995223 | 472,512        |                  13.91 | 4,793/s      |
|      26 |     5 | 2,5    |       5 |   5 |   64 |    9 | 0,0,0,4,0 |       8.99973e-05 |              0.0119427 |                0.0597134 | 291,520        |                  13.39 | 4,979/s      |
|      26 |     5 | 2,5    |       5 |   5 |   64 |    9 | 0,0,4,0,4 |       0.000111912 |             0.00995223 |                0.0597134 | 382,016        |                  12.28 | 5,427/s      |
|      26 |     5 | 2,5    |       5 |   5 |   64 |    9 | 0,4,0,4,0 |       7.71816e-05 |             0.00995223 |                0.0497611 | 382,016        |                  14.58 | 4,572/s      |
|      26 |     5 | 2,5    |       5 |   5 |   64 |    9 | 0,0,0,4,4 |       4.30281e-05 |             0.00597134 |                0.0298567 | 382,016        |                  11.7  | 5,699/s      |
|      26 |     5 | 2,5    |       5 |   5 |   64 |    9 | 0,0,4,4,0 |       1.58895e-05 |             0.00398089 |                0.0199045 | 382,016        |                  14.2  | 4,696/s      |
