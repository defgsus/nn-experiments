---
tags: ["kan"]
---

# Reviewing "KAE: Kolmogorov-Arnold Auto-Encoder for Representation Learning"

While browsing arxiv.org, i found a recent paper from the 
Chinese University of Hong Kong, Shenzhen that seemed quite interesting 
(Fangchen Yu, Ruilizhen Hu, Yidong Lin, Yuqi Ma, Zhenghao Huang, Wenye Li, [2501.00420](https://arxiv.org/abs/2501.00420)). 
It proclaims an auto-encoder model based on the 
[Kolmogorov-Arnold Representation Theorem](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Arnold_representation_theorem).

*Kolmogorov-Arnold Network* (KAN) is a relatively new approach to Neural Networks
([2404.19756](https://arxiv.org/abs/2404.19756v4)), where
activation functions are learned edge-wise instead of node-wise (or not at all). It's claimed
to have higher representational abilities compared to standard linear layers or 
[MLPs](https://en.wikipedia.org/wiki/Multilayer_perceptron). 

### Preface

Well, this is a little rant, which i add here without the intend to diminish the quality
of this particular paper. I actually like it.

Now, there is a, in my opinion, unhealthy trend in AI research to put the word *superior*
in the paper abstract and proclaim that the new model is superior to all state-of-the-art (SOTA)
models, or at least to the standard baselines. It might be true, but it might also just be true
for a very particular experimental setup and nowhere else.

There is another trend which is, to release the unreviewed preprint on arxiv.org and nowhere else.
I frequently stumble across papers which have, for example, tables with performance values, where
the text indicates that the proposed model is *superior*, while the actual numbers show the opposite.
Once you find such a detail, you realize that you have very likely wasted your time reading the
paper up to this point. I still wonder, why one would release such a paper. Does it provide a higher
academic ranking tying your name with the words superior and SOTA often enough?

I'm not a payed researcher so i generally have **no** access to Elsevier or Springer papers. 
Those papers are reviewed by researchers and, we all hope, none of those hoax papers make it
into a professional journal. The fact that reviewers are not payed for their reviewing work and
that professional publishers sell the access to those papers (at very high rates!), while the
research is generally funded by tax-payers money is the worst trend of all. 
<<<<<<< Updated upstream
=======
See, e.g., [here](https://www.lieffcabraser.com/antitrust/academic-journals/).
>>>>>>> Stashed changes

Anyways, that's just a side-note to provide some context. Out of curiosity, i will examine 
the paper myself. 

### Kolmogorov-Arnold Auto-Encoder

I like auto-encoders and did **a lot** of experiments with them. Since the authors of the KAE paper
provided their code ([github.com/SciYu/KAE](https://github.com/SciYu/KAE)), i naturally did 
not hesitate, copied the KAE model into my own experimental framework and tried a few things. 
The results, however, were chastening. The model is slower and does not perform nearly as good as 
my baseline CNN models. But, to be fair, it's only a single layer model by default. 
So... back to base research. 

Table 2 in the [paper](https://arxiv.org/abs/2501.00420), shows the *superiority* of the KAE model.
The table looks pretty convincing. And indeed, using the author's code, it can be completely 
reproduced. Generally, the paper is very clean and tidy and the conducted experiments are 
insightful. 

I cloned the repo (at commit `bce71dca` from Dec 31, 2024) and started a 
[jupyter lab](https://jupyter.org/) to reproduce the results of Table 2: 

```python
from typing import Optional
import math
import torch
import ExpToolKit


def run_experiment(
        config_path: str,
        num_trials: int = 10,
        is_print: bool = False,
        overrides: Optional[dict] = None,
):
    print(config_path)

    config = ExpToolKit.load_config(config_path)
    if overrides:
        for key, value in overrides.items():
            config[key].update(value)
        print("updated config:")
        display(config)
        print()

    test_losses = []
    for trial in range(num_trials):
        torch.cuda.empty_cache()

        config["TRAIN"]["random_seed"] = 2024 + trial
        train_setting = ExpToolKit.create_train_setting(config)
        if trial == 0:
            display(train_setting)
            num_params = sum(
                math.prod(p.shape)
                for p in train_setting["model"].parameters()
                if p.requires_grad
            )
            print(f"\nmodel parameters: {num_params:,}\n")

        print(f"trial {trial + 1}/{num_trials}")
        model, train_loss_epoch, train_loss_batch, epoch_time, test_loss_epoch = \
            ExpToolKit.train_and_test(**train_setting, is_print=is_print)

        print(f"test loss: {test_loss_epoch[-1]}, seconds: {epoch_time[-1]}")
        test_losses.append(test_loss_epoch[-1])

    print(f"average/best test loss: {sum(test_losses)/len(test_losses)} / {min(test_losses)}")
```

### Testing the baseline MLP model

The shipped configuration files enable testing all the models from Table 2 on 
l2 reconstruction loss of the MNIST dataset with a latent dimension of 16,
with Adam optimizer at learnrate 0.0001 and weight decay of 0.0001. In the paper, 
the baseline model is called `AE` but i switched to `MLP` for this article.

```python
run_experiment("model_config/config0.yaml")
```

```
model_config/config0.yaml

{'model': StandardAE(
   (encoder): Sequential(
     (0): DenseLayer(
       (layer): Linear(in_features=784, out_features=16, bias=True)
     )
     (1): ReLU(inplace=True)
   )
   (decoder): Sequential(
     (0): DenseLayer(
       (layer): Linear(in_features=16, out_features=784, bias=True)
     )
     (1): Sigmoid()
   )
 ),
 'train_loader': <torch.utils.data.dataloader.DataLoader at 0x7f0dcc999fd0>,
 'test_loader': <torch.utils.data.dataloader.DataLoader at 0x7f0dcc999f70>,
 'optimizer': Adam (
 Parameter Group 0
     amsgrad: False
     betas: (0.9, 0.999)
     capturable: False
     differentiable: False
     eps: 1e-08
     foreach: None
     fused: None
     lr: 0.0001
     maximize: False
     weight_decay: 0.0001
 ),
 'epochs': 10,
 'device': device(type='cuda'),
 'random_seed': 2024}

model parameters: 25,888

trial 1/10
test loss: 0.05721132168546319, seconds: 51.601099491119385
trial 2/10
test loss: 0.05495429076254368, seconds: 50.91784954071045
trial 3/10
test loss: 0.054549571592360735, seconds: 51.24936056137085
trial 4/10
test loss: 0.05354054784402251, seconds: 51.37986373901367
trial 5/10
test loss: 0.05453997133299708, seconds: 52.185614347457886
trial 6/10
test loss: 0.05438828486949206, seconds: 52.00752115249634
trial 7/10
test loss: 0.05443032365292311, seconds: 52.78800082206726
trial 8/10
test loss: 0.05106307221576571, seconds: 52.04966950416565
trial 9/10
test loss: 0.049927499424666164, seconds: 52.67065501213074
trial 10/10
test loss: 0.04967067549005151, seconds: 51.24051094055176

average/best test loss: 0.05342755588702858 / 0.04967067549005151
```

First of all, it's nice to see that my aged NVIDIA GeForce GTX 1660 Ti seems to be only twice
as slow as the NVIDIA TITAN V used by the authors. So i can do some test of my own.

The average test loss of **0.053** about matches the loss reported in Table 2 of **0.056 ±0.002**.
Note that i only ran the lr=0.0001 setup. The paper reports the average of all runs including
learnrate 0.00001 and two different settings of weight decay.


### Testing the KAE (p=3) model

```python
run_experiment("model_config/config6.yaml")
```

```
model_config/config6.yaml

{'model': StandardAE(
   (encoder): Sequential(
     (0): DenseLayer(
       (layer): KAELayer(order=3)
     )
     (1): ReLU(inplace=True)
   )
   (decoder): Sequential(
     (0): DenseLayer(
       (layer): KAELayer(order=3)
     )
     (1): Sigmoid()
   )
 ),
 'train_loader': <torch.utils.data.dataloader.DataLoader at 0x7f0ee02d3af0>,
 'test_loader': <torch.utils.data.dataloader.DataLoader at 0x7f0dc2b159d0>,
 'optimizer': Adam (
 Parameter Group 0
     amsgrad: False
     betas: (0.9, 0.999)
     capturable: False
     differentiable: False
     eps: 1e-08
     foreach: None
     fused: None
     lr: 0.0001
     maximize: False
     weight_decay: 0.0001
 ),
 'epochs': 10,
 'device': device(type='cuda'),
 'random_seed': 2024}

model parameters: 101,152

trial 1/10
test loss: 0.024415594851598145, seconds: 64.95739388465881
trial 2/10
test loss: 0.024201368540525438, seconds: 64.82365393638611
trial 3/10
test loss: 0.024129191506654026, seconds: 65.10156893730164
trial 4/10
test loss: 0.025636526104062796, seconds: 64.00057244300842
trial 5/10
test loss: 0.0220940746832639, seconds: 64.52272725105286
trial 6/10
test loss: 0.021915703685954212, seconds: 63.072824239730835
trial 7/10
test loss: 0.027073210990056395, seconds: 61.848424196243286
trial 8/10
test loss: 0.02452602991834283, seconds: 61.77449369430542
trial 9/10
test loss: 0.02409802973270416, seconds: 63.03647565841675
trial 10/10
test loss: 0.024723049299791456, seconds: 62.12168884277344

average/best test loss: 0.024281277931295336 / 0.021915703685954212
```

Test loss of **0.024** also matches the result in the paper. 
I also tested the other models and all reported results were reproduced. 

Now, there is an obvious detail that jumps out. The KAE model has 4 times the number of 
parameters as the MLP model. This is addressed in section 4.4 of the paper. The only way
to increase the number of parameters in the MLP autoencoder is to add a new layer. 

### Testing MLP with 64 hidden dims

The authors add a hidden layer with 64 dimensions, which increases the MLP model parameters
to 103,328, which almost exactly matches the KAE (p=3) model size.

(From now on, i spare you with the full textual output, which is listed in the 
[Appendix](#appendix))

```python
run_experiment("model_config/config0.yaml", overrides={"MODEL": {"hidden_dims": [64]}})
```

```
StandardAE(
   (encoder): Sequential(
     (0): DenseLayer(
       (layer): Linear(in_features=784, out_features=64, bias=True)
     )
     (1): ReLU(inplace=True)
     (2): DenseLayer(
       (layer): Linear(in_features=64, out_features=16, bias=True)
     )
     (3): ReLU(inplace=True)
   )
   (decoder): Sequential(
     (0): DenseLayer(
       (layer): Linear(in_features=16, out_features=64, bias=True)
     )
     (1): ReLU(inplace=True)
     (2): DenseLayer(
       (layer): Linear(in_features=64, out_features=784, bias=True)
     )
     (3): Sigmoid()
   )
 )
 
model parameters: 103,328

average/best test loss: 0.049578257398679854 / 0.047026059683412315
```

So, comparing the KAE and a similar sized MLP gives:

| model      |    params | test loss (10 runs) |
|------------|----------:|--------------------:|
| MLP (h=64) |   103,328 |               0.050 |
| KAE (p=3)  |   101,152 |               0.024 |


## Testing different activation functions

Whenever i see a *Sigmoid* activation i think about the good ol' Geoff Hinton times,
when this activation was used a lot. It puts everything between zero and one, which
intuitively seems to be a good choice for generating images.
However, personally, i never had a good experience with it.
The common activation function today is ReLU or some variant of it. For example, 
in this [MNIST autoencoder experiment](2023-11-12-mnist.md#varying-activation-function)
the best functions were ReLU6 and LeakyReLU.

The author's code does not allow setting up the activation functions in the config file
so i adjusted 
[the code](https://github.com/SciYu/KAE/blob/main/ExpToolKit/models.py#L156) 
to test different functions.

### MLP with ReLU6

Replacing both the ReLU and the Sigmoid with ReLU6:

```
model_config/config0.yaml

StandardAE(
   (encoder): Sequential(
     (0): DenseLayer(
       (layer): Linear(in_features=784, out_features=16, bias=True)
     )
     (1): ReLU6(inplace=True)
   )
   (decoder): Sequential(
     (0): DenseLayer(
       (layer): Linear(in_features=16, out_features=784, bias=True)
     )
     (1): ReLU6(inplace=True)
   )
)

average/best test loss: 0.03945172145031392 / 0.03638914376497269
```

It's significantly better than the original MLP from above. 
Adjusting the optimizer and the training batch size to my personal defaults, 
we can almost reach the KAE loss:

```python
run_experiment("model_config/config0.yaml", overrides={
    "TRAIN": {
        "batch_size": 64,
        "optim_type": "ADAMW",
        "lr": 0.0003, 
    }
})
```

```
average/best test loss: 0.029271703445987333 / 0.02781044684682682
```

Adding the 64-dim hidden layer to match the KAE model size, however, produces a 
slightly worse loss of 0.03.

Using the same optimizer and batch size settings for the KAE model, 
performance drops from 0.024 to 0.035. So that does not help. 

Switching the activation back to the original ReLU and Sigmoid but keeping the optimizer and
batch size only yields a test loss of 0.030.

Obviously, the authors have found good meta-parameters for training the KAE model. They just
don't hold for training the MLP.

I ran a couple more experiments but the reporting and the switching of activation functions 
got a bit difficult. So i [forked](https://github.com/defgsus/KAE) the repository 
and added the necessary code to run the experiments automatically. To reproduce the results:

```shell
git clone https://github.com/defgsus/KAE
# setup your virtualenv and install requirements.txt, torch and torchvision, then
python defgsus train
python defgsus test
```

The details are listed in the [Appendix](#appendix) (rendered with `python defgsus markdown`). 
Following is the table of all experiment results in compact form.


## Results of l2 reconstruction on MNIST

| model                                                                       | act           |    params | optim   |     lr |   batch size |                       test loss<br/>(10 runs)↓ |   train time<br/>(10 ep) |
|:----------------------------------------------------------------------------|:--------------|----------:|:--------|-------:|-------------:|-----------------------------------------------:|-------------------------:|
| [MLP](#mlp-relusigmoid-adam-lr00001-batch-size256)                          | relu/sigmoid  |    25,888 | Adam    | 0.0001 |          256 | 0.0532&nbsp;<span class="small">±0.0020</span> |                 48.4 sec |
| [MLP](#mlp-relu-adamw-lr00003-batch-size64)                                 | relu          |    25,888 | Adamw   | 0.0003 |           64 | 0.0294&nbsp;<span class="small">±0.0010</span> |                 55.2 sec |
| [MLP](#mlp-relu6-adam-lr00001-batch-size256)                                | relu6         |    25,888 | Adam    | 0.0001 |          256 | 0.0392&nbsp;<span class="small">±0.0015</span> |                 49.4 sec |
| [MLP](#mlp-relu6-adamw-lr00003-batch-size32)                                | relu6         |    25,888 | Adamw   | 0.0003 |           32 | 0.0298&nbsp;<span class="small">±0.0011</span> |                 64.1 sec |
| [MLP](#mlp-relu6-adamw-lr00003-batch-size64)                                | relu6         |    25,888 | Adamw   | 0.0003 |           64 | 0.0293&nbsp;<span class="small">±0.0010</span> |                 55.1 sec |
| [MLP](#mlp-relu6-adamw-lr00003-batch-size128)                               | relu6         |    25,888 | Adamw   | 0.0003 |          128 | 0.0304&nbsp;<span class="small">±0.0009</span> |                 51.1 sec |
| [MLP (hid=64)](#mlp-hid64-relusigmoid-adam-lr00001-batch-size256)           | relu/sigmoid  |   103,328 | Adam    | 0.0001 |          256 | 0.0496&nbsp;<span class="small">±0.0018</span> |                 50.1 sec |
| [MLP (hid=64)](#mlp-hid64-relu6-adamw-lr00003-batch-size64)                 | relu6         |   103,328 | Adamw   | 0.0003 |           64 | 0.0301&nbsp;<span class="small">±0.0015</span> |                 59.1 sec |
| [MLP (hid=128)](#mlp-hid128-relu6-adamw-lr00003-batch-size64)               | relu6         |   205,856 | Adamw   | 0.0003 |           64 | 0.0267&nbsp;<span class="small">±0.0013</span> |                 69.1 sec |
| [MLP (hid=256)](#mlp-hid256-relu6-adamw-lr00003-batch-size64)               | relu6         |   410,912 | Adamw   | 0.0003 |           64 | 0.0239&nbsp;<span class="small">±0.0011</span> |                 69.4 sec |
| [KAE (p=3)](#kae-p3-relusigmoid-adam-lr00001-batch-size256)                 | relu/sigmoid  |   101,152 | Adam    | 0.0001 |          256 | 0.0243&nbsp;<span class="small">±0.0014</span> |                 62.3 sec |
| [KAE (p=4)](#kae-p4-relusigmoid-adam-lr00001-batch-size256)                 | relu/sigmoid  |   126,240 | Adam    | 0.0001 |          256 | 0.0228&nbsp;<span class="small">±0.0011</span> |                 67.2 sec |
| [KAE (p=5)](#kae-p5-relusigmoid-adam-lr00001-batch-size256)                 | relu/sigmoid  |   151,328 | Adam    | 0.0001 |          256 | 0.0224&nbsp;<span class="small">±0.0009</span> |                 75.2 sec |
| [KAE (p=6)](#kae-p6-relusigmoid-adam-lr00001-batch-size256)                 | relu/sigmoid  |   176,416 | Adam    | 0.0001 |          256 | 0.0227&nbsp;<span class="small">±0.0011</span> |                 81.8 sec |
| [KAE (p=3)](#kae-p3-relu6sigmoid-adam-lr00001-batch-size256)                | relu6/sigmoid |   101,152 | Adam    | 0.0001 |          256 | 0.0235&nbsp;<span class="small">±0.0012</span> |                 63.7 sec |
| [KAE (p=3)](#kae-p3-relusigmoid-adam-lr00001-batch-size64)                  | relu/sigmoid  |   101,152 | Adam    | 0.0001 |           64 | 0.0256&nbsp;<span class="small">±0.0020</span> |                 72.5 sec |
| [KAE (p=3)](#kae-p3-relusigmoid-adam-lr00001-batch-size512)                 | relu/sigmoid  |   101,152 | Adam    | 0.0001 |          512 | 0.0255&nbsp;<span class="small">±0.0007</span> |                 61.5 sec |
| [KAE (p=3)](#kae-p3-relu6-adamw-lr00003-batch-size64)                       | relu6         |   101,152 | Adamw   | 0.0003 |           64 | 0.0346&nbsp;<span class="small">±0.0023</span> |                 70.8 sec |
| [KAE (p=3)](#kae-p3-relusigmoid-adamw-lr00003-batch-size64)                 | relu/sigmoid  |   101,152 | Adamw   | 0.0003 |           64 | 0.0308&nbsp;<span class="small">±0.0032</span> |                 70.5 sec |
| [KAE (hid=64, p=2)](#kae-hid64-p2-relusigmoid-adam-lr00001-batch-size256)   | relu/sigmoid  |   308,128 | Adam    | 0.0001 |          256 | 0.0256&nbsp;<span class="small">±0.0014</span> |                 94.6 sec |
| [KAE (hid=128, p=2)](#kae-hid128-p2-relusigmoid-adam-lr00001-batch-size256) | relu/sigmoid  |   615,456 | Adam    | 0.0001 |          256 | 0.0218&nbsp;<span class="small">±0.0015</span> |                157.9 sec |
| [KAE (hid=256, p=2)](#kae-hid256-p2-relusigmoid-adam-lr00001-batch-size256) | relu/sigmoid  | 1,230,112 | Adam    | 0.0001 |          256 | 0.0226&nbsp;<span class="small">±0.0029</span> |                250.1 sec |
| [KAE (hid=64, p=3)](#kae-hid64-p3-relusigmoid-adam-lr00001-batch-size256)   | relu/sigmoid  |   410,528 | Adam    | 0.0001 |          256 | 0.0222&nbsp;<span class="small">±0.0016</span> |                105.7 sec |
| [KAE (hid=128, p=3)](#kae-hid128-p3-relusigmoid-adam-lr00001-batch-size256) | relu/sigmoid  |   820,256 | Adam    | 0.0001 |          256 | 0.0176&nbsp;<span class="small">±0.0007</span> |                196.8 sec |
| [KAE (hid=256, p=3)](#kae-hid256-p3-relusigmoid-adam-lr00001-batch-size256) | relu/sigmoid  | 1,639,712 | Adam    | 0.0001 |          256 | 0.0159&nbsp;<span class="small">±0.0010</span> |                331.3 sec |
| [KAE (hid=64, p=4)](#kae-hid64-p4-relusigmoid-adam-lr00001-batch-size256)   | relu/sigmoid  |   512,928 | Adam    | 0.0001 |          256 | 0.0199&nbsp;<span class="small">±0.0019</span> |                142.4 sec |
| [KAE (hid=128, p=4)](#kae-hid128-p4-relusigmoid-adam-lr00001-batch-size256) | relu/sigmoid  | 1,025,056 | Adam    | 0.0001 |          256 | 0.0165&nbsp;<span class="small">±0.0008</span> |                258.2 sec |
| [KAE (hid=256, p=4)](#kae-hid256-p4-relusigmoid-adam-lr00001-batch-size256) | relu/sigmoid  | 2,049,312 | Adam    | 0.0001 |          256 | 0.0149&nbsp;<span class="small">±0.0006</span> |                437.4 sec |
| [KAE (hid=64, p=5)](#kae-hid64-p5-relusigmoid-adam-lr00001-batch-size256)   | relu/sigmoid  |   615,328 | Adam    | 0.0001 |          256 | 0.0182&nbsp;<span class="small">±0.0009</span> |                155.1 sec |
| [KAE (hid=128, p=5)](#kae-hid128-p5-relusigmoid-adam-lr00001-batch-size256) | relu/sigmoid  | 1,229,856 | Adam    | 0.0001 |          256 | 0.0155&nbsp;<span class="small">±0.0007</span> |                332.4 sec |
| [KAE (hid=256, p=5)](#kae-hid256-p5-relusigmoid-adam-lr00001-batch-size256) | relu/sigmoid  | 2,458,912 | Adam    | 0.0001 |          256 | 0.0141&nbsp;<span class="small">±0.0007</span> |                553.6 sec |

There surely is a way to increase the performance of the proposed KAE (p=3) model, with the right
batch size, optimizer settings, input and layer normalization or other means. 
I did not find it, yet, but adding an extra layer shows significant performance gains!

However, the difference of performance between a well-trained, much smaller 
MLP auto-encoder and the KAN auto-encoder (p=5) is only **0.007** in my experiments which does
not really justify using the term *superior* five times in the document. That might be 
my personal distaste but i rather would just term it *increased performance*.

Now, despite of all the numbers, what do the models actually do? They squeeze the 28x28 MNIST images
through a 16-dim latent vector and reproduce them. The compression ratio is 49! Let's see some
of the images from the MNIST validation set 
(odd columns are originals, even columns are reproductions):

| example images                                                                             |     loss | model                                                                                                                                            |
|:-------------------------------------------------------------------------------------------|---------:|:-------------------------------------------------------------------------------------------------------------------------------------------------|
| ![image](img/kae/reconstruction-mlp-relusigmoid-adam-lr00001-batch-size256.png)            |   0.0532 | MLP, relu/sigmoid, ADAM, lr=0.0001, batch size=256<br/><br/>The simple MLP used in Table 2 of the paper                                          |
| ![image](img/kae/reconstruction-mlp-relu6-adamw-lr00003-batch-size64.png)                  |   0.0293 | MLP, relu6, ADAMW, lr=0.0003, batch size=64<br/><br/>The improved MLP                                                                            |
| ![image](img/kae/reconstruction-kae-p3-relu6sigmoid-adam-lr00001-batch-size256.png)        |   0.0243 | KAE (p=3), relu/sigmoid, ADAM, lr=0.0001, batch size=256<br/><br/>Original KAE from paper                                                        |
| ![image](img/kae/reconstruction-kae-hid128-p2-relusigmoid-adam-lr00001-batch-size256.png)  |   0.0218 | KAE (hid=128, p=2), relu/sigmoid, ADAM, lr=0.0001, batch size=256<br/><br/>Improved KAE with hidden layer                                        |
| ![image](img/kae/reconstruction-kae-hid256-p5-relusigmoid-adam-lr00001-batch-size256.png)  |   0.0141 | KAE (hid=256, p=5), relu/sigmoid, ADAM, lr=0.0001, batch size=256<br/><br/>Best model in these experiments,<br/>although completely oversized ;) |



## Results of l2 reconstruction on MNIST including extra tasks

Below is the same table including the test results for **classification**, **retrieval** and 
**denoising**, as detailed in the author's
[README](https://github.com/SciYu/KAE/blob/main/README.md#evaluate-model-on-various-tasks) file.
(You can click the headers to sort the table)

| model                                                                       | act           |    params | optim/lr/bs      |                       test loss<br/>(10 runs)↓ |   train time<br/>(10 ep) | classifier<br/>accuracy↑ | retriever<br/>recall@5↑ | denoiser<br/>salt&pepper↓ |
|:----------------------------------------------------------------------------|:--------------|----------:|:-----------------|-----------------------------------------------:|-------------------------:|-------------------------:|------------------------:|--------------------------:|
| [MLP](#mlp-relusigmoid-adam-lr00001-batch-size256)                          | relu/sigmoid  |    25,888 | Adam/0.0001/256  | 0.0532&nbsp;<span class="small">±0.0020</span> |                 48.4 sec |                   0.8859 |                  0.4021 |                    0.0870 |
| [MLP](#mlp-relu-adamw-lr00003-batch-size64)                                 | relu          |    25,888 | Adamw/0.0003/64  | 0.0294&nbsp;<span class="small">±0.0010</span> |                 55.2 sec |                   0.9524 |                  0.5261 |                    0.0750 |
| [MLP](#mlp-relu6-adam-lr00001-batch-size256)                                | relu6         |    25,888 | Adam/0.0001/256  | 0.0392&nbsp;<span class="small">±0.0015</span> |                 49.4 sec |                   0.9266 |                  0.5021 |                    0.0772 |
| [MLP](#mlp-relu6-adamw-lr00003-batch-size32)                                | relu6         |    25,888 | Adamw/0.0003/32  | 0.0298&nbsp;<span class="small">±0.0011</span> |                 64.1 sec |                   0.9515 |                  0.5407 |                    0.0762 |
| [MLP](#mlp-relu6-adamw-lr00003-batch-size64)                                | relu6         |    25,888 | Adamw/0.0003/64  | 0.0293&nbsp;<span class="small">±0.0010</span> |                 55.1 sec |                   0.9523 |                  0.5296 |                    0.0750 |
| [MLP](#mlp-relu6-adamw-lr00003-batch-size128)                               | relu6         |    25,888 | Adamw/0.0003/128 | 0.0304&nbsp;<span class="small">±0.0009</span> |                 51.1 sec |                   0.9492 |                  0.5157 |                    0.0742 |
| [MLP (hid=64)](#mlp-hid64-relusigmoid-adam-lr00001-batch-size256)           | relu/sigmoid  |   103,328 | Adam/0.0001/256  | 0.0496&nbsp;<span class="small">±0.0018</span> |                 50.1 sec |                   0.8084 |                  0.3531 |                    0.0851 |
| [MLP (hid=64)](#mlp-hid64-relu6-adamw-lr00003-batch-size64)                 | relu6         |   103,328 | Adamw/0.0003/64  | 0.0301&nbsp;<span class="small">±0.0015</span> |                 59.1 sec |                   0.9392 |                  0.5348 |                    0.0746 |
| [MLP (hid=128)](#mlp-hid128-relu6-adamw-lr00003-batch-size64)               | relu6         |   205,856 | Adamw/0.0003/64  | 0.0267&nbsp;<span class="small">±0.0013</span> |                 69.1 sec |                   0.9468 |                  0.5641 |                    0.0726 |
| [MLP (hid=256)](#mlp-hid256-relu6-adamw-lr00003-batch-size64)               | relu6         |   410,912 | Adamw/0.0003/64  | 0.0239&nbsp;<span class="small">±0.0011</span> |                 69.4 sec |                   0.9581 |                  0.6025 |                    0.0714 |
| [KAE (p=3)](#kae-p3-relusigmoid-adam-lr00001-batch-size256)                 | relu/sigmoid  |   101,152 | Adam/0.0001/256  | 0.0243&nbsp;<span class="small">±0.0014</span> |                 62.3 sec |                   0.9523 |                  0.5446 |                    0.0672 |
| [KAE (p=4)](#kae-p4-relusigmoid-adam-lr00001-batch-size256)                 | relu/sigmoid  |   126,240 | Adam/0.0001/256  | 0.0228&nbsp;<span class="small">±0.0011</span> |                 67.2 sec |                   0.9540 |                  0.5375 |                    0.0681 |
| [KAE (p=5)](#kae-p5-relusigmoid-adam-lr00001-batch-size256)                 | relu/sigmoid  |   151,328 | Adam/0.0001/256  | 0.0224&nbsp;<span class="small">±0.0009</span> |                 75.2 sec |                   0.9533 |                  0.5375 |                    0.0702 |
| [KAE (p=6)](#kae-p6-relusigmoid-adam-lr00001-batch-size256)                 | relu/sigmoid  |   176,416 | Adam/0.0001/256  | 0.0227&nbsp;<span class="small">±0.0011</span> |                 81.8 sec |                   0.9541 |                  0.5608 |                    0.0704 |
| [KAE (p=3)](#kae-p3-relu6sigmoid-adam-lr00001-batch-size256)                | relu6/sigmoid |   101,152 | Adam/0.0001/256  | 0.0235&nbsp;<span class="small">±0.0012</span> |                 63.7 sec |                   0.9592 |                  0.6092 |                    0.0665 |
| [KAE (p=3)](#kae-p3-relusigmoid-adam-lr00001-batch-size64)                  | relu/sigmoid  |   101,152 | Adam/0.0001/64   | 0.0256&nbsp;<span class="small">±0.0020</span> |                 72.5 sec |                   0.9530 |                  0.5546 |                    0.0676 |
| [KAE (p=3)](#kae-p3-relusigmoid-adam-lr00001-batch-size512)                 | relu/sigmoid  |   101,152 | Adam/0.0001/512  | 0.0255&nbsp;<span class="small">±0.0007</span> |                 61.5 sec |                   0.9399 |                  0.5172 |                    0.0688 |
| [KAE (p=3)](#kae-p3-relu6-adamw-lr00003-batch-size64)                       | relu6         |   101,152 | Adamw/0.0003/64  | 0.0346&nbsp;<span class="small">±0.0023</span> |                 70.8 sec |                   0.9365 |                  0.5522 |                    0.0887 |
| [KAE (p=3)](#kae-p3-relusigmoid-adamw-lr00003-batch-size64)                 | relu/sigmoid  |   101,152 | Adamw/0.0003/64  | 0.0308&nbsp;<span class="small">±0.0032</span> |                 70.5 sec |                   0.9370 |                  0.5342 |                    0.0788 |
| [KAE (hid=64, p=2)](#kae-hid64-p2-relusigmoid-adam-lr00001-batch-size256)   | relu/sigmoid  |   308,128 | Adam/0.0001/256  | 0.0256&nbsp;<span class="small">±0.0014</span> |                 94.6 sec |                   0.9339 |                  0.5227 |                    0.0692 |
| [KAE (hid=128, p=2)](#kae-hid128-p2-relusigmoid-adam-lr00001-batch-size256) | relu/sigmoid  |   615,456 | Adam/0.0001/256  | 0.0218&nbsp;<span class="small">±0.0015</span> |                157.9 sec |                   0.9527 |                  0.5730 |                    0.0649 |
| [KAE (hid=256, p=2)](#kae-hid256-p2-relusigmoid-adam-lr00001-batch-size256) | relu/sigmoid  | 1,230,112 | Adam/0.0001/256  | 0.0226&nbsp;<span class="small">±0.0029</span> |                250.1 sec |                   0.9511 |                  0.5753 |                    0.0652 |
| [KAE (hid=64, p=3)](#kae-hid64-p3-relusigmoid-adam-lr00001-batch-size256)   | relu/sigmoid  |   410,528 | Adam/0.0001/256  | 0.0222&nbsp;<span class="small">±0.0016</span> |                105.7 sec |                   0.9529 |                  0.5846 |                    0.0706 |
| [KAE (hid=128, p=3)](#kae-hid128-p3-relusigmoid-adam-lr00001-batch-size256) | relu/sigmoid  |   820,256 | Adam/0.0001/256  | 0.0176&nbsp;<span class="small">±0.0007</span> |                196.8 sec |                   0.9542 |                  0.6200 |                    0.0684 |
| [KAE (hid=256, p=3)](#kae-hid256-p3-relusigmoid-adam-lr00001-batch-size256) | relu/sigmoid  | 1,639,712 | Adam/0.0001/256  | 0.0159&nbsp;<span class="small">±0.0010</span> |                331.3 sec |                   0.9582 |                  0.6357 |                    0.0656 |
| [KAE (hid=64, p=4)](#kae-hid64-p4-relusigmoid-adam-lr00001-batch-size256)   | relu/sigmoid  |   512,928 | Adam/0.0001/256  | 0.0199&nbsp;<span class="small">±0.0019</span> |                142.4 sec |                   0.9587 |                  0.6048 |                    0.0717 |
| [KAE (hid=128, p=4)](#kae-hid128-p4-relusigmoid-adam-lr00001-batch-size256) | relu/sigmoid  | 1,025,056 | Adam/0.0001/256  | 0.0165&nbsp;<span class="small">±0.0008</span> |                258.2 sec |                   0.9592 |                  0.6278 |                    0.0676 |
| [KAE (hid=256, p=4)](#kae-hid256-p4-relusigmoid-adam-lr00001-batch-size256) | relu/sigmoid  | 2,049,312 | Adam/0.0001/256  | 0.0149&nbsp;<span class="small">±0.0006</span> |                437.4 sec |                   0.9608 |                  0.6482 |                    0.0668 |
| [KAE (hid=64, p=5)](#kae-hid64-p5-relusigmoid-adam-lr00001-batch-size256)   | relu/sigmoid  |   615,328 | Adam/0.0001/256  | 0.0182&nbsp;<span class="small">±0.0009</span> |                155.1 sec |                   0.9566 |                  0.6170 |                    0.0698 |
| [KAE (hid=128, p=5)](#kae-hid128-p5-relusigmoid-adam-lr00001-batch-size256) | relu/sigmoid  | 1,229,856 | Adam/0.0001/256  | 0.0155&nbsp;<span class="small">±0.0007</span> |                332.4 sec |                   0.9607 |                  0.6241 |                    0.0671 |
| [KAE (hid=256, p=5)](#kae-hid256-p5-relusigmoid-adam-lr00001-batch-size256) | relu/sigmoid  | 2,458,912 | Adam/0.0001/256  | 0.0141&nbsp;<span class="small">±0.0007</span> |                553.6 sec |                   0.9612 |                  0.6313 |                    0.0675 |

In conclusion i would argue that this particular polynomial KAN-based autoencoder is 
an interesting new approach. The model code is easy to read and certainly invites 
for further experimentation.


# Appendix

Just listing the lengthy experiment setups / outputs here. It's not much to see, just a
proof of reproducibility. You can repeat the experiments with

```shell
git clone https://github.com/defgsus/KAE
# setup your virtualenv and install requirements.txt, torch and torchvision, then
python defgsus train
python defgsus test
# render table and output
python defgsus markdown
```

It takes a couple of hours, though!


### MLP, relu/sigmoid, ADAM, lr=0.0001, batch size=256

```
model_config/config0.yaml

updated config:
{'DATA': {'type': 'MNIST'},
 'MODEL': {'hidden_dims': [],
           'latent_dim': 16,
           'layer_type': 'LINEAR',
           'model_type': 'AE'},
 'TRAIN': {'batch_size': 256,
           'epochs': 10,
           'lr': 0.0001,
           'optim_type': 'ADAM',
           'random_seed': 2024,
           'weight_decay': 0.0001}}

{'device': device(type='cuda'),
 'epochs': 10,
 'model': StandardAE(
  (encoder): Sequential(
    (0): DenseLayer(
      (layer): Linear(in_features=784, out_features=16, bias=True)
    )
    (1): ReLU(inplace=True)
  )
  (decoder): Sequential(
    (0): DenseLayer(
      (layer): Linear(in_features=16, out_features=784, bias=True)
    )
    (1): Sigmoid()
  )
),
 'optimizer': Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 0.0001
    maximize: False
    weight_decay: 0.0001
),
 'random_seed': 2024,
 'test_loader': <torch.utils.data.dataloader.DataLoader object at 0x7f36ef6afa90>,
 'train_loader': <torch.utils.data.dataloader.DataLoader object at 0x7f36ef6af1f0>}

model parameters: 25,888

trial 1/10
test loss: 0.054844805505126715, seconds: 48.74358034133911
trial 2/10
test loss: 0.05495429076254368, seconds: 47.40274238586426
trial 3/10
test loss: 0.054549571592360735, seconds: 48.13005805015564
trial 4/10
test loss: 0.05354054784402251, seconds: 47.44930839538574
trial 5/10
test loss: 0.05453997133299708, seconds: 47.6246395111084
trial 6/10
test loss: 0.05438828486949206, seconds: 47.376731872558594
trial 7/10
test loss: 0.05443032365292311, seconds: 48.409247636795044
trial 8/10
test loss: 0.05106307221576571, seconds: 47.933077812194824
trial 9/10
test loss: 0.049927499424666164, seconds: 50.86521673202515
trial 10/10
test loss: 0.04967067549005151, seconds: 50.02432680130005

average/best test loss: 0.053190904268994935 / 0.04967067549005151

```


### MLP, relu, ADAMW, lr=0.0003, batch size=64

```
model_config/config0.yaml

updated config:
{'DATA': {'type': 'MNIST'},
 'MODEL': {'activation': 'relu',
           'hidden_dims': [],
           'latent_dim': 16,
           'layer_type': 'LINEAR',
           'model_type': 'AE'},
 'TRAIN': {'batch_size': 64,
           'epochs': 10,
           'lr': 0.0003,
           'optim_type': 'ADAMW',
           'random_seed': 2024,
           'weight_decay': 0.0001}}

{'device': device(type='cuda'),
 'epochs': 10,
 'model': StandardAE(
  (encoder): Sequential(
    (0): DenseLayer(
      (layer): Linear(in_features=784, out_features=16, bias=True)
    )
    (1): ReLU(inplace=True)
  )
  (decoder): Sequential(
    (0): DenseLayer(
      (layer): Linear(in_features=16, out_features=784, bias=True)
    )
    (1): ReLU(inplace=True)
  )
),
 'optimizer': AdamW (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 0.0003
    maximize: False
    weight_decay: 0.0001
),
 'random_seed': 2024,
 'test_loader': <torch.utils.data.dataloader.DataLoader object at 0x7f37d1351cd0>,
 'train_loader': <torch.utils.data.dataloader.DataLoader object at 0x7f36ef6af880>}

model parameters: 25,888

trial 1/10
test loss: 0.029944378443679233, seconds: 55.40973496437073
trial 2/10
test loss: 0.027822728844205287, seconds: 54.660828590393066
trial 3/10
test loss: 0.03028506580384294, seconds: 54.57886362075806
trial 4/10
test loss: 0.029726570009425947, seconds: 54.942195415496826
trial 5/10
test loss: 0.028753836965484985, seconds: 55.46592450141907
trial 6/10
test loss: 0.02827958514688501, seconds: 54.46421766281128
trial 7/10
test loss: 0.03000373778876605, seconds: 55.709019899368286
trial 8/10
test loss: 0.029764349957939924, seconds: 54.3574538230896
trial 9/10
test loss: 0.028051003173088573, seconds: 56.52020335197449
trial 10/10
test loss: 0.03092550927666342, seconds: 55.56819820404053

average/best test loss: 0.02935567654099814 / 0.027822728844205287

```


### MLP, relu6, ADAM, lr=0.0001, batch size=256

```
model_config/config0.yaml

updated config:
{'DATA': {'type': 'MNIST'},
 'MODEL': {'activation': 'relu6',
           'hidden_dims': [],
           'latent_dim': 16,
           'layer_type': 'LINEAR',
           'model_type': 'AE'},
 'TRAIN': {'batch_size': 256,
           'epochs': 10,
           'lr': 0.0001,
           'optim_type': 'ADAM',
           'random_seed': 2024,
           'weight_decay': 0.0001}}

{'device': device(type='cuda'),
 'epochs': 10,
 'model': StandardAE(
  (encoder): Sequential(
    (0): DenseLayer(
      (layer): Linear(in_features=784, out_features=16, bias=True)
    )
    (1): ReLU6(inplace=True)
  )
  (decoder): Sequential(
    (0): DenseLayer(
      (layer): Linear(in_features=16, out_features=784, bias=True)
    )
    (1): ReLU6(inplace=True)
  )
),
 'optimizer': Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 0.0001
    maximize: False
    weight_decay: 0.0001
),
 'random_seed': 2024,
 'test_loader': <torch.utils.data.dataloader.DataLoader object at 0x7f37d1351760>,
 'train_loader': <torch.utils.data.dataloader.DataLoader object at 0x7f36ef6afe20>}

model parameters: 25,888

trial 1/10
test loss: 0.03937941854819656, seconds: 49.730987310409546
trial 2/10
test loss: 0.03887385474517942, seconds: 48.26471924781799
trial 3/10
test loss: 0.03937122141942382, seconds: 48.90603280067444
trial 4/10
test loss: 0.04118271768093109, seconds: 48.22850513458252
trial 5/10
test loss: 0.03768241703510285, seconds: 49.30234146118164
trial 6/10
test loss: 0.03638914376497269, seconds: 50.181209087371826
trial 7/10
test loss: 0.04111840622499585, seconds: 50.58647346496582
trial 8/10
test loss: 0.03880140176042914, seconds: 50.34295725822449
trial 9/10
test loss: 0.03865907033905387, seconds: 48.52420377731323
trial 10/10
test loss: 0.04104078523814678, seconds: 49.796762466430664

average/best test loss: 0.03924984367564321 / 0.03638914376497269

```


### MLP, relu6, ADAMW, lr=0.0003, batch size=32

```
model_config/config0.yaml

updated config:
{'DATA': {'type': 'MNIST'},
 'MODEL': {'activation': 'relu6',
           'hidden_dims': [],
           'latent_dim': 16,
           'layer_type': 'LINEAR',
           'model_type': 'AE'},
 'TRAIN': {'batch_size': 32,
           'epochs': 10,
           'lr': 0.0003,
           'optim_type': 'ADAMW',
           'random_seed': 2024,
           'weight_decay': 0.0001}}

{'device': device(type='cuda'),
 'epochs': 10,
 'model': StandardAE(
  (encoder): Sequential(
    (0): DenseLayer(
      (layer): Linear(in_features=784, out_features=16, bias=True)
    )
    (1): ReLU6(inplace=True)
  )
  (decoder): Sequential(
    (0): DenseLayer(
      (layer): Linear(in_features=16, out_features=784, bias=True)
    )
    (1): ReLU6(inplace=True)
  )
),
 'optimizer': AdamW (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 0.0003
    maximize: False
    weight_decay: 0.0001
),
 'random_seed': 2024,
 'test_loader': <torch.utils.data.dataloader.DataLoader object at 0x7f36ef6afdc0>,
 'train_loader': <torch.utils.data.dataloader.DataLoader object at 0x7f37d1351d90>}

model parameters: 25,888

trial 1/10
test loss: 0.030145346975555053, seconds: 63.46911311149597
trial 2/10
test loss: 0.028754215681562407, seconds: 65.45769238471985
trial 3/10
test loss: 0.03130609097595984, seconds: 63.62731218338013
trial 4/10
test loss: 0.02995197093501068, seconds: 63.822922229766846
trial 5/10
test loss: 0.028888749393125693, seconds: 65.16088700294495
trial 6/10
test loss: 0.027876560507824246, seconds: 64.96088886260986
trial 7/10
test loss: 0.030093948038431784, seconds: 63.83512210845947
trial 8/10
test loss: 0.030830478617034782, seconds: 63.844937801361084
trial 9/10
test loss: 0.02906663065996414, seconds: 63.632938861846924
trial 10/10
test loss: 0.030966824111037742, seconds: 63.29023098945618

average/best test loss: 0.029788081589550635 / 0.027876560507824246

```


### MLP, relu6, ADAMW, lr=0.0003, batch size=64

```
model_config/config0.yaml

updated config:
{'DATA': {'type': 'MNIST'},
 'MODEL': {'activation': 'relu6',
           'hidden_dims': [],
           'latent_dim': 16,
           'layer_type': 'LINEAR',
           'model_type': 'AE'},
 'TRAIN': {'batch_size': 64,
           'epochs': 10,
           'lr': 0.0003,
           'optim_type': 'ADAMW',
           'random_seed': 2024,
           'weight_decay': 0.0001}}

{'device': device(type='cuda'),
 'epochs': 10,
 'model': StandardAE(
  (encoder): Sequential(
    (0): DenseLayer(
      (layer): Linear(in_features=784, out_features=16, bias=True)
    )
    (1): ReLU6(inplace=True)
  )
  (decoder): Sequential(
    (0): DenseLayer(
      (layer): Linear(in_features=16, out_features=784, bias=True)
    )
    (1): ReLU6(inplace=True)
  )
),
 'optimizer': AdamW (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 0.0003
    maximize: False
    weight_decay: 0.0001
),
 'random_seed': 2024,
 'test_loader': <torch.utils.data.dataloader.DataLoader object at 0x7f36ef6affd0>,
 'train_loader': <torch.utils.data.dataloader.DataLoader object at 0x7f37d13516a0>}

model parameters: 25,888

trial 1/10
test loss: 0.029864317351940332, seconds: 55.62854290008545
trial 2/10
test loss: 0.02781044684682682, seconds: 54.88072180747986
trial 3/10
test loss: 0.030311387293278032, seconds: 53.7932653427124
trial 4/10
test loss: 0.029717421194740162, seconds: 55.48655819892883
trial 5/10
test loss: 0.02875516924318994, seconds: 55.250420808792114
trial 6/10
test loss: 0.028260746875860887, seconds: 55.32818794250488
trial 7/10
test loss: 0.029995871423061485, seconds: 54.1175856590271
trial 8/10
test loss: 0.02970408286401041, seconds: 55.907835483551025
trial 9/10
test loss: 0.02802042883767444, seconds: 55.862600564956665
trial 10/10
test loss: 0.03092010900568051, seconds: 55.07930874824524

average/best test loss: 0.029335998093626296 / 0.02781044684682682

```


### MLP, relu6, ADAMW, lr=0.0003, batch size=128

```
model_config/config0.yaml

updated config:
{'DATA': {'type': 'MNIST'},
 'MODEL': {'activation': 'relu6',
           'hidden_dims': [],
           'latent_dim': 16,
           'layer_type': 'LINEAR',
           'model_type': 'AE'},
 'TRAIN': {'batch_size': 128,
           'epochs': 10,
           'lr': 0.0003,
           'optim_type': 'ADAMW',
           'random_seed': 2024,
           'weight_decay': 0.0001}}

{'device': device(type='cuda'),
 'epochs': 10,
 'model': StandardAE(
  (encoder): Sequential(
    (0): DenseLayer(
      (layer): Linear(in_features=784, out_features=16, bias=True)
    )
    (1): ReLU6(inplace=True)
  )
  (decoder): Sequential(
    (0): DenseLayer(
      (layer): Linear(in_features=16, out_features=784, bias=True)
    )
    (1): ReLU6(inplace=True)
  )
),
 'optimizer': AdamW (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 0.0003
    maximize: False
    weight_decay: 0.0001
),
 'random_seed': 2024,
 'test_loader': <torch.utils.data.dataloader.DataLoader object at 0x7f37d10cc640>,
 'train_loader': <torch.utils.data.dataloader.DataLoader object at 0x7f37d10cc280>}

model parameters: 25,888

trial 1/10
test loss: 0.030623565604792364, seconds: 51.72880673408508
trial 2/10
test loss: 0.02996114940865885, seconds: 51.676554679870605
trial 3/10
test loss: 0.03071878032310854, seconds: 50.302478313446045
trial 4/10
test loss: 0.02936490311558488, seconds: 49.930197954177856
trial 5/10
test loss: 0.029585700600018985, seconds: 50.63589954376221
trial 6/10
test loss: 0.02945860029681574, seconds: 52.13498044013977
trial 7/10
test loss: 0.03143414425887639, seconds: 50.9267463684082
trial 8/10
test loss: 0.03150970132762118, seconds: 49.932409048080444
trial 9/10
test loss: 0.02962162767690194, seconds: 51.81868243217468
trial 10/10
test loss: 0.032096313666316524, seconds: 51.64334535598755

average/best test loss: 0.030437448627869547 / 0.02936490311558488

```


### MLP (hid=64), relu/sigmoid, ADAM, lr=0.0001, batch size=256

```
model_config/config0.yaml

updated config:
{'DATA': {'type': 'MNIST'},
 'MODEL': {'hidden_dims': [64],
           'latent_dim': 16,
           'layer_type': 'LINEAR',
           'model_type': 'AE'},
 'TRAIN': {'batch_size': 256,
           'epochs': 10,
           'lr': 0.0001,
           'optim_type': 'ADAM',
           'random_seed': 2024,
           'weight_decay': 0.0001}}

{'device': device(type='cuda'),
 'epochs': 10,
 'model': StandardAE(
  (encoder): Sequential(
    (0): DenseLayer(
      (layer): Linear(in_features=784, out_features=64, bias=True)
    )
    (1): ReLU(inplace=True)
    (2): DenseLayer(
      (layer): Linear(in_features=64, out_features=16, bias=True)
    )
    (3): ReLU(inplace=True)
  )
  (decoder): Sequential(
    (0): DenseLayer(
      (layer): Linear(in_features=16, out_features=64, bias=True)
    )
    (1): ReLU(inplace=True)
    (2): DenseLayer(
      (layer): Linear(in_features=64, out_features=784, bias=True)
    )
    (3): Sigmoid()
  )
),
 'optimizer': Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 0.0001
    maximize: False
    weight_decay: 0.0001
),
 'random_seed': 2024,
 'test_loader': <torch.utils.data.dataloader.DataLoader object at 0x7f37d1351cd0>,
 'train_loader': <torch.utils.data.dataloader.DataLoader object at 0x7f37d10cc040>}

model parameters: 103,328

trial 1/10
test loss: 0.051934999600052836, seconds: 50.947391748428345
trial 2/10
test loss: 0.04881709320470691, seconds: 48.80104160308838
trial 3/10
test loss: 0.0477697504684329, seconds: 50.15599513053894
trial 4/10
test loss: 0.04974388424307108, seconds: 51.431276082992554
trial 5/10
test loss: 0.048030237294733526, seconds: 51.55111837387085
trial 6/10
test loss: 0.04976865621283651, seconds: 48.58534121513367
trial 7/10
test loss: 0.047026059683412315, seconds: 49.52819037437439
trial 8/10
test loss: 0.05237543722614646, seconds: 51.40358114242554
trial 9/10
test loss: 0.04865111084654927, seconds: 48.65099883079529
trial 10/10
test loss: 0.05166534520685673, seconds: 49.85130286216736

average/best test loss: 0.049578257398679854 / 0.047026059683412315

```


### MLP (hid=64), relu6, ADAMW, lr=0.0003, batch size=64

```
model_config/config0.yaml

updated config:
{'DATA': {'type': 'MNIST'},
 'MODEL': {'activation': 'relu6',
           'hidden_dims': [64],
           'latent_dim': 16,
           'layer_type': 'LINEAR',
           'model_type': 'AE'},
 'TRAIN': {'batch_size': 64,
           'epochs': 10,
           'lr': 0.0003,
           'optim_type': 'ADAMW',
           'random_seed': 2024,
           'weight_decay': 0.0001}}

{'device': device(type='cuda'),
 'epochs': 10,
 'model': StandardAE(
  (encoder): Sequential(
    (0): DenseLayer(
      (layer): Linear(in_features=784, out_features=64, bias=True)
    )
    (1): ReLU6(inplace=True)
    (2): DenseLayer(
      (layer): Linear(in_features=64, out_features=16, bias=True)
    )
    (3): ReLU6(inplace=True)
  )
  (decoder): Sequential(
    (0): DenseLayer(
      (layer): Linear(in_features=16, out_features=64, bias=True)
    )
    (1): ReLU6(inplace=True)
    (2): DenseLayer(
      (layer): Linear(in_features=64, out_features=784, bias=True)
    )
    (3): ReLU6(inplace=True)
  )
),
 'optimizer': AdamW (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 0.0003
    maximize: False
    weight_decay: 0.0001
),
 'random_seed': 2024,
 'test_loader': <torch.utils.data.dataloader.DataLoader object at 0x7f37d10cc5b0>,
 'train_loader': <torch.utils.data.dataloader.DataLoader object at 0x7f37d10ccac0>}

model parameters: 103,328

trial 1/10
test loss: 0.031225492110013204, seconds: 58.489842653274536
trial 2/10
test loss: 0.03038751133450657, seconds: 57.93046164512634
trial 3/10
test loss: 0.030448248562444546, seconds: 59.022109031677246
trial 4/10
test loss: 0.03345142389131579, seconds: 59.17180943489075
trial 5/10
test loss: 0.0289119388789508, seconds: 59.28048276901245
trial 6/10
test loss: 0.03027447387813383, seconds: 59.327950954437256
trial 7/10
test loss: 0.028075553085299056, seconds: 59.53023934364319
trial 8/10
test loss: 0.029163281296848493, seconds: 60.20081067085266
trial 9/10
test loss: 0.03053849844178956, seconds: 60.13843059539795
trial 10/10
test loss: 0.028196997003285748, seconds: 58.36703062057495

average/best test loss: 0.030067341848258766 / 0.028075553085299056

```


### MLP (hid=128), relu6, ADAMW, lr=0.0003, batch size=64

```
model_config/config0.yaml

updated config:
{'DATA': {'type': 'MNIST'},
 'MODEL': {'activation': 'relu6',
           'hidden_dims': [128],
           'latent_dim': 16,
           'layer_type': 'LINEAR',
           'model_type': 'AE'},
 'TRAIN': {'batch_size': 64,
           'epochs': 10,
           'lr': 0.0003,
           'optim_type': 'ADAMW',
           'random_seed': 2024,
           'weight_decay': 0.0001}}

{'device': device(type='cuda'),
 'epochs': 10,
 'model': StandardAE(
  (encoder): Sequential(
    (0): DenseLayer(
      (layer): Linear(in_features=784, out_features=128, bias=True)
    )
    (1): ReLU6(inplace=True)
    (2): DenseLayer(
      (layer): Linear(in_features=128, out_features=16, bias=True)
    )
    (3): ReLU6(inplace=True)
  )
  (decoder): Sequential(
    (0): DenseLayer(
      (layer): Linear(in_features=16, out_features=128, bias=True)
    )
    (1): ReLU6(inplace=True)
    (2): DenseLayer(
      (layer): Linear(in_features=128, out_features=784, bias=True)
    )
    (3): ReLU6(inplace=True)
  )
),
 'optimizer': AdamW (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 0.0003
    maximize: False
    weight_decay: 0.0001
),
 'random_seed': 2024,
 'test_loader': <torch.utils.data.dataloader.DataLoader object at 0x7f8baf281970>,
 'train_loader': <torch.utils.data.dataloader.DataLoader object at 0x7f8baf2816d0>}

model parameters: 205,856

trial 1/10
test loss: 0.02757667852150407, seconds: 70.6561987400055
trial 2/10
test loss: 0.029030630614157695, seconds: 66.79325604438782
trial 3/10
test loss: 0.025214034945342193, seconds: 67.67058849334717
trial 4/10
test loss: 0.026811098003653205, seconds: 71.8734381198883
trial 5/10
test loss: 0.02679019872170345, seconds: 73.1500997543335
trial 6/10
test loss: 0.024743097069062244, seconds: 68.46631598472595
trial 7/10
test loss: 0.02485356943764884, seconds: 68.00167441368103
trial 8/10
test loss: 0.027746190607642673, seconds: 69.04577732086182
trial 9/10
test loss: 0.026676727375786774, seconds: 69.02845931053162
trial 10/10
test loss: 0.02724222306185847, seconds: 65.86859202384949

average/best test loss: 0.02666844483583596 / 0.024743097069062244

```


### MLP (hid=256), relu6, ADAMW, lr=0.0003, batch size=64

```
model_config/config0.yaml

updated config:
{'DATA': {'type': 'MNIST'},
 'MODEL': {'activation': 'relu6',
           'hidden_dims': [256],
           'latent_dim': 16,
           'layer_type': 'LINEAR',
           'model_type': 'AE'},
 'TRAIN': {'batch_size': 64,
           'epochs': 10,
           'lr': 0.0003,
           'optim_type': 'ADAMW',
           'random_seed': 2024,
           'weight_decay': 0.0001}}

{'device': device(type='cuda'),
 'epochs': 10,
 'model': StandardAE(
  (encoder): Sequential(
    (0): DenseLayer(
      (layer): Linear(in_features=784, out_features=256, bias=True)
    )
    (1): ReLU6(inplace=True)
    (2): DenseLayer(
      (layer): Linear(in_features=256, out_features=16, bias=True)
    )
    (3): ReLU6(inplace=True)
  )
  (decoder): Sequential(
    (0): DenseLayer(
      (layer): Linear(in_features=16, out_features=256, bias=True)
    )
    (1): ReLU6(inplace=True)
    (2): DenseLayer(
      (layer): Linear(in_features=256, out_features=784, bias=True)
    )
    (3): ReLU6(inplace=True)
  )
),
 'optimizer': AdamW (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 0.0003
    maximize: False
    weight_decay: 0.0001
),
 'random_seed': 2024,
 'test_loader': <torch.utils.data.dataloader.DataLoader object at 0x7f8baf281130>,
 'train_loader': <torch.utils.data.dataloader.DataLoader object at 0x7f8baf281a60>}

model parameters: 410,912

trial 1/10
test loss: 0.02382377028512727, seconds: 67.96561050415039
trial 2/10
test loss: 0.024177915114126387, seconds: 66.49497652053833
trial 3/10
test loss: 0.0236373558687936, seconds: 67.77597880363464
trial 4/10
test loss: 0.02233525288475167, seconds: 67.86813306808472
trial 5/10
test loss: 0.026097432799210216, seconds: 69.33182644844055
trial 6/10
test loss: 0.025345706956306842, seconds: 72.02309012413025
trial 7/10
test loss: 0.02403836791065468, seconds: 70.12623643875122
trial 8/10
test loss: 0.02219296034401769, seconds: 68.07407331466675
trial 9/10
test loss: 0.023282034619218985, seconds: 71.18422937393188
trial 10/10
test loss: 0.024291369779284592, seconds: 72.9444727897644

average/best test loss: 0.023922216656149194 / 0.02219296034401769

```


### KAE (p=3), relu/sigmoid, ADAM, lr=0.0001, batch size=256

```
model_config/config6.yaml

updated config:
{'DATA': {'type': 'MNIST'},
 'MODEL': {'hidden_dims': [],
           'latent_dim': 16,
           'layer_type': 'KAE',
           'model_type': 'AE',
           'order': 3},
 'TRAIN': {'batch_size': 256,
           'epochs': 10,
           'lr': 0.0001,
           'optim_type': 'ADAM',
           'random_seed': 2024,
           'weight_decay': 0.0001}}

{'device': device(type='cuda'),
 'epochs': 10,
 'model': StandardAE(
  (encoder): Sequential(
    (0): DenseLayer(
      (layer): KAELayer(order=3)
    )
    (1): ReLU(inplace=True)
  )
  (decoder): Sequential(
    (0): DenseLayer(
      (layer): KAELayer(order=3)
    )
    (1): Sigmoid()
  )
),
 'optimizer': Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 0.0001
    maximize: False
    weight_decay: 0.0001
),
 'random_seed': 2024,
 'test_loader': <torch.utils.data.dataloader.DataLoader object at 0x7f37d10cc550>,
 'train_loader': <torch.utils.data.dataloader.DataLoader object at 0x7f37d10ccdf0>}

model parameters: 101,152

trial 1/10
test loss: 0.024415594851598145, seconds: 61.92895293235779
trial 2/10
test loss: 0.024201368540525438, seconds: 60.32642364501953
trial 3/10
test loss: 0.024129191506654026, seconds: 62.38642168045044
trial 4/10
test loss: 0.025636526104062796, seconds: 61.56795644760132
trial 5/10
test loss: 0.0220940746832639, seconds: 62.635332107543945
trial 6/10
test loss: 0.021915703685954212, seconds: 62.71131467819214
trial 7/10
test loss: 0.027073210990056395, seconds: 63.939212799072266
trial 8/10
test loss: 0.02452602991834283, seconds: 63.21239233016968
trial 9/10
test loss: 0.02409802973270416, seconds: 62.5829131603241
trial 10/10
test loss: 0.024723049299791456, seconds: 61.32445979118347

average/best test loss: 0.024281277931295336 / 0.021915703685954212

```


### KAE (p=4), relu/sigmoid, ADAM, lr=0.0001, batch size=256

```
model_config/config6.yaml

updated config:
{'DATA': {'type': 'MNIST'},
 'MODEL': {'hidden_dims': [],
           'latent_dim': 16,
           'layer_type': 'KAE',
           'model_type': 'AE',
           'order': 4},
 'TRAIN': {'batch_size': 256,
           'epochs': 10,
           'lr': 0.0001,
           'optim_type': 'ADAM',
           'random_seed': 2024,
           'weight_decay': 0.0001}}

{'device': device(type='cuda'),
 'epochs': 10,
 'model': StandardAE(
  (encoder): Sequential(
    (0): DenseLayer(
      (layer): KAELayer(order=4)
    )
    (1): ReLU(inplace=True)
  )
  (decoder): Sequential(
    (0): DenseLayer(
      (layer): KAELayer(order=4)
    )
    (1): Sigmoid()
  )
),
 'optimizer': Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 0.0001
    maximize: False
    weight_decay: 0.0001
),
 'random_seed': 2024,
 'test_loader': <torch.utils.data.dataloader.DataLoader object at 0x7f37d10cc3a0>,
 'train_loader': <torch.utils.data.dataloader.DataLoader object at 0x7f36ef6af9a0>}

model parameters: 126,240

trial 1/10
test loss: 0.0213471460621804, seconds: 65.92803621292114
trial 2/10
test loss: 0.022648265538737177, seconds: 67.24404215812683
trial 3/10
test loss: 0.022140852641314268, seconds: 68.13891673088074
trial 4/10
test loss: 0.02260885932482779, seconds: 67.84793710708618
trial 5/10
test loss: 0.024192711059004068, seconds: 67.24179553985596
trial 6/10
test loss: 0.02392191574908793, seconds: 66.79944491386414
trial 7/10
test loss: 0.024204003950580956, seconds: 68.41916704177856
trial 8/10
test loss: 0.02161008436232805, seconds: 67.58119535446167
trial 9/10
test loss: 0.023785065673291684, seconds: 65.27455425262451
trial 10/10
test loss: 0.021693919878453018, seconds: 67.33229947090149

average/best test loss: 0.022815282423980534 / 0.0213471460621804

```


### KAE (p=5), relu/sigmoid, ADAM, lr=0.0001, batch size=256

```
model_config/config6.yaml

updated config:
{'DATA': {'type': 'MNIST'},
 'MODEL': {'hidden_dims': [],
           'latent_dim': 16,
           'layer_type': 'KAE',
           'model_type': 'AE',
           'order': 5},
 'TRAIN': {'batch_size': 256,
           'epochs': 10,
           'lr': 0.0001,
           'optim_type': 'ADAM',
           'random_seed': 2024,
           'weight_decay': 0.0001}}

{'device': device(type='cuda'),
 'epochs': 10,
 'model': StandardAE(
  (encoder): Sequential(
    (0): DenseLayer(
      (layer): KAELayer(order=5)
    )
    (1): ReLU(inplace=True)
  )
  (decoder): Sequential(
    (0): DenseLayer(
      (layer): KAELayer(order=5)
    )
    (1): Sigmoid()
  )
),
 'optimizer': Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 0.0001
    maximize: False
    weight_decay: 0.0001
),
 'random_seed': 2024,
 'test_loader': <torch.utils.data.dataloader.DataLoader object at 0x7f37d10cce80>,
 'train_loader': <torch.utils.data.dataloader.DataLoader object at 0x7f37d1351a90>}

model parameters: 151,328

trial 1/10
test loss: 0.023980671400204302, seconds: 74.61662101745605
trial 2/10
test loss: 0.022810124233365058, seconds: 74.86650276184082
trial 3/10
test loss: 0.021235586237162353, seconds: 74.01571750640869
trial 4/10
test loss: 0.02169646918773651, seconds: 74.23548221588135
trial 5/10
test loss: 0.02105184900574386, seconds: 75.06123161315918
trial 6/10
test loss: 0.023493063217028976, seconds: 76.60210585594177
trial 7/10
test loss: 0.023077776981517674, seconds: 75.59740281105042
trial 8/10
test loss: 0.02147274692542851, seconds: 74.46156001091003
trial 9/10
test loss: 0.022462127590551974, seconds: 76.81111788749695
trial 10/10
test loss: 0.022454827837646008, seconds: 75.66343092918396

average/best test loss: 0.022373524261638522 / 0.02105184900574386

```


### KAE (p=6), relu/sigmoid, ADAM, lr=0.0001, batch size=256

```
model_config/config6.yaml

updated config:
{'DATA': {'type': 'MNIST'},
 'MODEL': {'hidden_dims': [],
           'latent_dim': 16,
           'layer_type': 'KAE',
           'model_type': 'AE',
           'order': 6},
 'TRAIN': {'batch_size': 256,
           'epochs': 10,
           'lr': 0.0001,
           'optim_type': 'ADAM',
           'random_seed': 2024,
           'weight_decay': 0.0001}}

{'device': device(type='cuda'),
 'epochs': 10,
 'model': StandardAE(
  (encoder): Sequential(
    (0): DenseLayer(
      (layer): KAELayer(order=6)
    )
    (1): ReLU(inplace=True)
  )
  (decoder): Sequential(
    (0): DenseLayer(
      (layer): KAELayer(order=6)
    )
    (1): Sigmoid()
  )
),
 'optimizer': Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 0.0001
    maximize: False
    weight_decay: 0.0001
),
 'random_seed': 2024,
 'test_loader': <torch.utils.data.dataloader.DataLoader object at 0x7f37d10cc7c0>,
 'train_loader': <torch.utils.data.dataloader.DataLoader object at 0x7f36ef6afd60>}

model parameters: 176,416

trial 1/10
test loss: 0.02248598770238459, seconds: 79.81454825401306
trial 2/10
test loss: 0.022188912145793438, seconds: 79.89576363563538
trial 3/10
test loss: 0.024314309703186154, seconds: 81.67849159240723
trial 4/10
test loss: 0.024558632262051107, seconds: 81.62779688835144
trial 5/10
test loss: 0.02227715775370598, seconds: 81.00719881057739
trial 6/10
test loss: 0.02273458130657673, seconds: 83.1842896938324
trial 7/10
test loss: 0.023906563920900226, seconds: 81.46982002258301
trial 8/10
test loss: 0.02215076144784689, seconds: 83.68938827514648
trial 9/10
test loss: 0.021122285490855576, seconds: 83.52689623832703
trial 10/10
test loss: 0.021565554104745387, seconds: 81.79217624664307

average/best test loss: 0.02273047458380461 / 0.021122285490855576

```


### KAE (p=3), relu6/sigmoid, ADAM, lr=0.0001, batch size=256

```
model_config/config6.yaml

updated config:
{'DATA': {'type': 'MNIST'},
 'MODEL': {'activation': ['relu6', 'sigmoid'],
           'hidden_dims': [],
           'latent_dim': 16,
           'layer_type': 'KAE',
           'model_type': 'AE',
           'order': 3},
 'TRAIN': {'batch_size': 256,
           'epochs': 10,
           'lr': 0.0001,
           'optim_type': 'ADAM',
           'random_seed': 2024,
           'weight_decay': 0.0001}}

{'device': device(type='cuda'),
 'epochs': 10,
 'model': StandardAE(
  (encoder): Sequential(
    (0): DenseLayer(
      (layer): KAELayer(order=3)
    )
    (1): ReLU6(inplace=True)
  )
  (decoder): Sequential(
    (0): DenseLayer(
      (layer): KAELayer(order=3)
    )
    (1): Sigmoid()
  )
),
 'optimizer': Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 0.0001
    maximize: False
    weight_decay: 0.0001
),
 'random_seed': 2024,
 'test_loader': <torch.utils.data.dataloader.DataLoader object at 0x7f37d10ccbe0>,
 'train_loader': <torch.utils.data.dataloader.DataLoader object at 0x7f36ef6afdc0>}

model parameters: 101,152

trial 1/10
test loss: 0.02309356229379773, seconds: 64.85483431816101
trial 2/10
test loss: 0.02193830031901598, seconds: 64.11653161048889
trial 3/10
test loss: 0.024399833753705025, seconds: 65.46109867095947
trial 4/10
test loss: 0.023626095009967686, seconds: 64.1012167930603
trial 5/10
test loss: 0.02228605728596449, seconds: 64.01207900047302
trial 6/10
test loss: 0.022312369430437684, seconds: 63.67721509933472
trial 7/10
test loss: 0.025964177446439862, seconds: 63.69110441207886
trial 8/10
test loss: 0.023402246087789534, seconds: 61.47948455810547
trial 9/10
test loss: 0.02332183890976012, seconds: 63.08373951911926
trial 10/10
test loss: 0.025107008311897515, seconds: 62.94281458854675

average/best test loss: 0.023545148884877562 / 0.02193830031901598

```


### KAE (p=3), relu/sigmoid, ADAM, lr=0.0001, batch size=64

```
model_config/config6.yaml

updated config:
{'DATA': {'type': 'MNIST'},
 'MODEL': {'hidden_dims': [],
           'latent_dim': 16,
           'layer_type': 'KAE',
           'model_type': 'AE',
           'order': 3},
 'TRAIN': {'batch_size': 64,
           'epochs': 10,
           'lr': 0.0001,
           'optim_type': 'ADAM',
           'random_seed': 2024,
           'weight_decay': 0.0001}}

{'device': device(type='cuda'),
 'epochs': 10,
 'model': StandardAE(
  (encoder): Sequential(
    (0): DenseLayer(
      (layer): KAELayer(order=3)
    )
    (1): ReLU(inplace=True)
  )
  (decoder): Sequential(
    (0): DenseLayer(
      (layer): KAELayer(order=3)
    )
    (1): Sigmoid()
  )
),
 'optimizer': Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 0.0001
    maximize: False
    weight_decay: 0.0001
),
 'random_seed': 2024,
 'test_loader': <torch.utils.data.dataloader.DataLoader object at 0x7f37d10cc310>,
 'train_loader': <torch.utils.data.dataloader.DataLoader object at 0x7f36ef6afd90>}

model parameters: 101,152

trial 1/10
test loss: 0.02615715488554186, seconds: 72.68000745773315
trial 2/10
test loss: 0.02310845555417287, seconds: 71.651535987854
trial 3/10
test loss: 0.02798851725354696, seconds: 71.62791204452515
trial 4/10
test loss: 0.027769463587623493, seconds: 73.73475170135498
trial 5/10
test loss: 0.023415053381946434, seconds: 72.88235187530518
trial 6/10
test loss: 0.02224666504248692, seconds: 73.22373843193054
trial 7/10
test loss: 0.027987814644814295, seconds: 73.20678901672363
trial 8/10
test loss: 0.02645594084481145, seconds: 73.04287147521973
trial 9/10
test loss: 0.026078407195912805, seconds: 73.70239424705505
trial 10/10
test loss: 0.02469016543951384, seconds: 69.29945802688599

average/best test loss: 0.025589763783037095 / 0.02224666504248692

```


### KAE (p=3), relu/sigmoid, ADAM, lr=0.0001, batch size=512

```
model_config/config6.yaml

updated config:
{'DATA': {'type': 'MNIST'},
 'MODEL': {'hidden_dims': [],
           'latent_dim': 16,
           'layer_type': 'KAE',
           'model_type': 'AE',
           'order': 3},
 'TRAIN': {'batch_size': 512,
           'epochs': 10,
           'lr': 0.0001,
           'optim_type': 'ADAM',
           'random_seed': 2024,
           'weight_decay': 0.0001}}

{'device': device(type='cuda'),
 'epochs': 10,
 'model': StandardAE(
  (encoder): Sequential(
    (0): DenseLayer(
      (layer): KAELayer(order=3)
    )
    (1): ReLU(inplace=True)
  )
  (decoder): Sequential(
    (0): DenseLayer(
      (layer): KAELayer(order=3)
    )
    (1): Sigmoid()
  )
),
 'optimizer': Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 0.0001
    maximize: False
    weight_decay: 0.0001
),
 'random_seed': 2024,
 'test_loader': <torch.utils.data.dataloader.DataLoader object at 0x7f37d10cce20>,
 'train_loader': <torch.utils.data.dataloader.DataLoader object at 0x7f36ef6afb80>}

model parameters: 101,152

trial 1/10
test loss: 0.025707779359072445, seconds: 59.64279365539551
trial 2/10
test loss: 0.02481404719874263, seconds: 60.674640417099
trial 3/10
test loss: 0.026527704764157535, seconds: 60.42444372177124
trial 4/10
test loss: 0.025667520519346, seconds: 62.54549741744995
trial 5/10
test loss: 0.025927974842488766, seconds: 61.43470597267151
trial 6/10
test loss: 0.02496118126437068, seconds: 63.49837613105774
trial 7/10
test loss: 0.026538813766092063, seconds: 62.94175696372986
trial 8/10
test loss: 0.024276717472821473, seconds: 62.804282665252686
trial 9/10
test loss: 0.025847604498267174, seconds: 60.39290261268616
trial 10/10
test loss: 0.02517965892329812, seconds: 60.71731376647949

average/best test loss: 0.025544900260865692 / 0.024276717472821473

```


### KAE (p=3), relu6, ADAMW, lr=0.0003, batch size=64

```
model_config/config6.yaml

updated config:
{'DATA': {'type': 'MNIST'},
 'MODEL': {'activation': ['relu6', 'relu6'],
           'hidden_dims': [],
           'latent_dim': 16,
           'layer_type': 'KAE',
           'model_type': 'AE',
           'order': 3},
 'TRAIN': {'batch_size': 64,
           'epochs': 10,
           'lr': 0.0003,
           'optim_type': 'ADAMW',
           'random_seed': 2024,
           'weight_decay': 0.0001}}

{'device': device(type='cuda'),
 'epochs': 10,
 'model': StandardAE(
  (encoder): Sequential(
    (0): DenseLayer(
      (layer): KAELayer(order=3)
    )
    (1): ReLU6(inplace=True)
  )
  (decoder): Sequential(
    (0): DenseLayer(
      (layer): KAELayer(order=3)
    )
    (1): ReLU6(inplace=True)
  )
),
 'optimizer': AdamW (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 0.0003
    maximize: False
    weight_decay: 0.0001
),
 'random_seed': 2024,
 'test_loader': <torch.utils.data.dataloader.DataLoader object at 0x7f37d1351c70>,
 'train_loader': <torch.utils.data.dataloader.DataLoader object at 0x7f37d1351130>}

model parameters: 101,152

trial 1/10
test loss: 0.03023416113559228, seconds: 70.26620602607727
trial 2/10
test loss: 0.039044849622021816, seconds: 71.62893223762512
trial 3/10
test loss: 0.0326307240375288, seconds: 71.98599314689636
trial 4/10
test loss: 0.03546572939320734, seconds: 70.04723477363586
trial 5/10
test loss: 0.03381784916351176, seconds: 70.74350953102112
trial 6/10
test loss: 0.03386436209414795, seconds: 70.8341600894928
trial 7/10
test loss: 0.03602883121247884, seconds: 70.6436402797699
trial 8/10
test loss: 0.036401670034618895, seconds: 70.8380868434906
trial 9/10
test loss: 0.035208017191594575, seconds: 70.97638368606567
trial 10/10
test loss: 0.03290520119629088, seconds: 69.85479497909546

average/best test loss: 0.034560139508099316 / 0.03023416113559228

```


### KAE (p=3), relu/sigmoid, ADAMW, lr=0.0003, batch size=64

```
model_config/config6.yaml

updated config:
{'DATA': {'type': 'MNIST'},
 'MODEL': {'hidden_dims': [],
           'latent_dim': 16,
           'layer_type': 'KAE',
           'model_type': 'AE',
           'order': 3},
 'TRAIN': {'batch_size': 64,
           'epochs': 10,
           'lr': 0.0003,
           'optim_type': 'ADAMW',
           'random_seed': 2024,
           'weight_decay': 0.0001}}

{'device': device(type='cuda'),
 'epochs': 10,
 'model': StandardAE(
  (encoder): Sequential(
    (0): DenseLayer(
      (layer): KAELayer(order=3)
    )
    (1): ReLU(inplace=True)
  )
  (decoder): Sequential(
    (0): DenseLayer(
      (layer): KAELayer(order=3)
    )
    (1): Sigmoid()
  )
),
 'optimizer': AdamW (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 0.0003
    maximize: False
    weight_decay: 0.0001
),
 'random_seed': 2024,
 'test_loader': <torch.utils.data.dataloader.DataLoader object at 0x7f37d1351e80>,
 'train_loader': <torch.utils.data.dataloader.DataLoader object at 0x7f37d13512b0>}

model parameters: 101,152

trial 1/10
test loss: 0.027002361751380998, seconds: 69.63095498085022
trial 2/10
test loss: 0.03218813594074765, seconds: 68.59184980392456
trial 3/10
test loss: 0.0328096655571157, seconds: 71.18175435066223
trial 4/10
test loss: 0.03489779873163837, seconds: 70.70620441436768
trial 5/10
test loss: 0.026414862899169042, seconds: 70.2962257862091
trial 6/10
test loss: 0.02531469639414435, seconds: 70.63081979751587
trial 7/10
test loss: 0.03441416723712994, seconds: 69.97365498542786
trial 8/10
test loss: 0.030439759812252536, seconds: 71.21950697898865
trial 9/10
test loss: 0.03216738462637944, seconds: 70.41167783737183
trial 10/10
test loss: 0.032150394182391226, seconds: 72.3699688911438

average/best test loss: 0.030779922713234924 / 0.02531469639414435

```


### KAE (hid=64, p=2), relu/sigmoid, ADAM, lr=0.0001, batch size=256

```
model_config/config6.yaml

updated config:
{'DATA': {'type': 'MNIST'},
 'MODEL': {'hidden_dims': [64],
           'latent_dim': 16,
           'layer_type': 'KAE',
           'model_type': 'AE',
           'order': 2},
 'TRAIN': {'batch_size': 256,
           'epochs': 10,
           'lr': 0.0001,
           'optim_type': 'ADAM',
           'random_seed': 2024,
           'weight_decay': 0.0001}}

{'device': device(type='cuda'),
 'epochs': 10,
 'model': StandardAE(
  (encoder): Sequential(
    (0): DenseLayer(
      (layer): KAELayer(order=2)
    )
    (1): ReLU(inplace=True)
    (2): DenseLayer(
      (layer): KAELayer(order=2)
    )
    (3): ReLU(inplace=True)
  )
  (decoder): Sequential(
    (0): DenseLayer(
      (layer): KAELayer(order=2)
    )
    (1): ReLU(inplace=True)
    (2): DenseLayer(
      (layer): KAELayer(order=2)
    )
    (3): Sigmoid()
  )
),
 'optimizer': Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 0.0001
    maximize: False
    weight_decay: 0.0001
),
 'random_seed': 2024,
 'test_loader': <torch.utils.data.dataloader.DataLoader object at 0x7fb46abbdbe0>,
 'train_loader': <torch.utils.data.dataloader.DataLoader object at 0x7fb46abbdc10>}

model parameters: 308,128

trial 1/10
test loss: 0.024810216249898077, seconds: 87.20655822753906
trial 2/10
test loss: 0.025054814480245113, seconds: 90.82882761955261
trial 3/10
test loss: 0.029328040825203062, seconds: 93.76674461364746
trial 4/10
test loss: 0.026678384700790047, seconds: 94.57724380493164
trial 5/10
test loss: 0.024223004141822456, seconds: 94.48876214027405
trial 6/10
test loss: 0.024324131943285466, seconds: 93.90495753288269
trial 7/10
test loss: 0.02501974389888346, seconds: 96.939692735672
trial 8/10
test loss: 0.026064124144613742, seconds: 96.95923495292664
trial 9/10
test loss: 0.025096864346414803, seconds: 98.49881863594055
trial 10/10
test loss: 0.025496953958645464, seconds: 98.96546983718872

average/best test loss: 0.02560962786898017 / 0.024223004141822456

```


### KAE (hid=128, p=2), relu/sigmoid, ADAM, lr=0.0001, batch size=256

```
model_config/config6.yaml

updated config:
{'DATA': {'type': 'MNIST'},
 'MODEL': {'hidden_dims': [128],
           'latent_dim': 16,
           'layer_type': 'KAE',
           'model_type': 'AE',
           'order': 2},
 'TRAIN': {'batch_size': 256,
           'epochs': 10,
           'lr': 0.0001,
           'optim_type': 'ADAM',
           'random_seed': 2024,
           'weight_decay': 0.0001}}

{'device': device(type='cuda'),
 'epochs': 10,
 'model': StandardAE(
  (encoder): Sequential(
    (0): DenseLayer(
      (layer): KAELayer(order=2)
    )
    (1): ReLU(inplace=True)
    (2): DenseLayer(
      (layer): KAELayer(order=2)
    )
    (3): ReLU(inplace=True)
  )
  (decoder): Sequential(
    (0): DenseLayer(
      (layer): KAELayer(order=2)
    )
    (1): ReLU(inplace=True)
    (2): DenseLayer(
      (layer): KAELayer(order=2)
    )
    (3): Sigmoid()
  )
),
 'optimizer': Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 0.0001
    maximize: False
    weight_decay: 0.0001
),
 'random_seed': 2024,
 'test_loader': <torch.utils.data.dataloader.DataLoader object at 0x7fb549e842b0>,
 'train_loader': <torch.utils.data.dataloader.DataLoader object at 0x7fb46abbd9d0>}

model parameters: 615,456

trial 1/10
test loss: 0.02336266408674419, seconds: 158.22575211524963
trial 2/10
test loss: 0.019091003900393845, seconds: 158.271062374115
trial 3/10
test loss: 0.019837975315749646, seconds: 159.3715991973877
trial 4/10
test loss: 0.02341140890493989, seconds: 155.35668230056763
trial 5/10
test loss: 0.02207847419194877, seconds: 152.05787658691406
trial 6/10
test loss: 0.020837245509028435, seconds: 158.38157606124878
trial 7/10
test loss: 0.02213256466202438, seconds: 162.78448700904846
trial 8/10
test loss: 0.020774537371471523, seconds: 161.09157872200012
trial 9/10
test loss: 0.023839181195944546, seconds: 156.59118175506592
trial 10/10
test loss: 0.022777973441407084, seconds: 157.24475932121277

average/best test loss: 0.02181430285796523 / 0.019091003900393845

```


### KAE (hid=256, p=2), relu/sigmoid, ADAM, lr=0.0001, batch size=256

```
model_config/config6.yaml

updated config:
{'DATA': {'type': 'MNIST'},
 'MODEL': {'hidden_dims': [256],
           'latent_dim': 16,
           'layer_type': 'KAE',
           'model_type': 'AE',
           'order': 2},
 'TRAIN': {'batch_size': 256,
           'epochs': 10,
           'lr': 0.0001,
           'optim_type': 'ADAM',
           'random_seed': 2024,
           'weight_decay': 0.0001}}

{'device': device(type='cuda'),
 'epochs': 10,
 'model': StandardAE(
  (encoder): Sequential(
    (0): DenseLayer(
      (layer): KAELayer(order=2)
    )
    (1): ReLU(inplace=True)
    (2): DenseLayer(
      (layer): KAELayer(order=2)
    )
    (3): ReLU(inplace=True)
  )
  (decoder): Sequential(
    (0): DenseLayer(
      (layer): KAELayer(order=2)
    )
    (1): ReLU(inplace=True)
    (2): DenseLayer(
      (layer): KAELayer(order=2)
    )
    (3): Sigmoid()
  )
),
 'optimizer': Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 0.0001
    maximize: False
    weight_decay: 0.0001
),
 'random_seed': 2024,
 'test_loader': <torch.utils.data.dataloader.DataLoader object at 0x7fb46abbdd90>,
 'train_loader': <torch.utils.data.dataloader.DataLoader object at 0x7fb46abbd490>}

model parameters: 1,230,112

trial 1/10
test loss: 0.026064584124833347, seconds: 250.39859056472778
trial 2/10
test loss: 0.024486870411783455, seconds: 249.52806568145752
trial 3/10
test loss: 0.02816329142078757, seconds: 250.88722562789917
trial 4/10
test loss: 0.01849869654979557, seconds: 250.4723780155182
trial 5/10
test loss: 0.023125345911830665, seconds: 250.70067381858826
trial 6/10
test loss: 0.022905606171116234, seconds: 246.0978467464447
trial 7/10
test loss: 0.021652239840477705, seconds: 250.5683355331421
trial 8/10
test loss: 0.019134251540526746, seconds: 251.5078740119934
trial 9/10
test loss: 0.021603700052946807, seconds: 250.71569180488586
trial 10/10
test loss: 0.020234312349930405, seconds: 250.3537197113037

average/best test loss: 0.02258688983740285 / 0.01849869654979557

```


### KAE (hid=64, p=3), relu/sigmoid, ADAM, lr=0.0001, batch size=256

```
model_config/config6.yaml

updated config:
{'DATA': {'type': 'MNIST'},
 'MODEL': {'hidden_dims': [64],
           'latent_dim': 16,
           'layer_type': 'KAE',
           'model_type': 'AE',
           'order': 3},
 'TRAIN': {'batch_size': 256,
           'epochs': 10,
           'lr': 0.0001,
           'optim_type': 'ADAM',
           'random_seed': 2024,
           'weight_decay': 0.0001}}

{'device': device(type='cuda'),
 'epochs': 10,
 'model': StandardAE(
  (encoder): Sequential(
    (0): DenseLayer(
      (layer): KAELayer(order=3)
    )
    (1): ReLU(inplace=True)
    (2): DenseLayer(
      (layer): KAELayer(order=3)
    )
    (3): ReLU(inplace=True)
  )
  (decoder): Sequential(
    (0): DenseLayer(
      (layer): KAELayer(order=3)
    )
    (1): ReLU(inplace=True)
    (2): DenseLayer(
      (layer): KAELayer(order=3)
    )
    (3): Sigmoid()
  )
),
 'optimizer': Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 0.0001
    maximize: False
    weight_decay: 0.0001
),
 'random_seed': 2024,
 'test_loader': <torch.utils.data.dataloader.DataLoader object at 0x7fa16d6daee0>,
 'train_loader': <torch.utils.data.dataloader.DataLoader object at 0x7fa16d6da130>}

model parameters: 410,528

trial 1/10
test loss: 0.022235007444396614, seconds: 101.52159070968628
trial 2/10
test loss: 0.022399124642834067, seconds: 102.52322673797607
trial 3/10
test loss: 0.02662791032344103, seconds: 105.62324666976929
trial 4/10
test loss: 0.022848214395344256, seconds: 104.23346161842346
trial 5/10
test loss: 0.02125693657435477, seconds: 107.69089484214783
trial 6/10
test loss: 0.021431715320795776, seconds: 106.03088760375977
trial 7/10
test loss: 0.02070729318074882, seconds: 107.04297685623169
trial 8/10
test loss: 0.022259943094104527, seconds: 106.59215521812439
trial 9/10
test loss: 0.021174417017027734, seconds: 106.26171398162842
trial 10/10
test loss: 0.021046917978674175, seconds: 109.13478350639343

average/best test loss: 0.022198747997172176 / 0.02070729318074882

```


### KAE (hid=128, p=3), relu/sigmoid, ADAM, lr=0.0001, batch size=256

```
model_config/config6.yaml

updated config:
{'DATA': {'type': 'MNIST'},
 'MODEL': {'hidden_dims': [128],
           'latent_dim': 16,
           'layer_type': 'KAE',
           'model_type': 'AE',
           'order': 3},
 'TRAIN': {'batch_size': 256,
           'epochs': 10,
           'lr': 0.0001,
           'optim_type': 'ADAM',
           'random_seed': 2024,
           'weight_decay': 0.0001}}

{'device': device(type='cuda'),
 'epochs': 10,
 'model': StandardAE(
  (encoder): Sequential(
    (0): DenseLayer(
      (layer): KAELayer(order=3)
    )
    (1): ReLU(inplace=True)
    (2): DenseLayer(
      (layer): KAELayer(order=3)
    )
    (3): ReLU(inplace=True)
  )
  (decoder): Sequential(
    (0): DenseLayer(
      (layer): KAELayer(order=3)
    )
    (1): ReLU(inplace=True)
    (2): DenseLayer(
      (layer): KAELayer(order=3)
    )
    (3): Sigmoid()
  )
),
 'optimizer': Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 0.0001
    maximize: False
    weight_decay: 0.0001
),
 'random_seed': 2024,
 'test_loader': <torch.utils.data.dataloader.DataLoader object at 0x7f5243ed50a0>,
 'train_loader': <torch.utils.data.dataloader.DataLoader object at 0x7f516422b250>}

model parameters: 820,256

trial 1/10
test loss: 0.018272698298096655, seconds: 189.71533584594727
trial 2/10
test loss: 0.018411319074220955, seconds: 190.0505406856537
trial 3/10
test loss: 0.01766556976363063, seconds: 206.22142052650452
trial 4/10
test loss: 0.01832906869240105, seconds: 196.5188329219818
trial 5/10
test loss: 0.01683416806627065, seconds: 196.67028522491455
trial 6/10
test loss: 0.016869719396345316, seconds: 202.8109085559845
trial 7/10
test loss: 0.016718478570692242, seconds: 199.81436681747437
trial 8/10
test loss: 0.016854225541464984, seconds: 194.35895204544067
trial 9/10
test loss: 0.018173357425257563, seconds: 194.3624300956726
trial 10/10
test loss: 0.018231959571130572, seconds: 197.3516390323639

average/best test loss: 0.017636056439951062 / 0.016718478570692242

```


### KAE (hid=256, p=3), relu/sigmoid, ADAM, lr=0.0001, batch size=256

```
model_config/config6.yaml

updated config:
{'DATA': {'type': 'MNIST'},
 'MODEL': {'hidden_dims': [256],
           'latent_dim': 16,
           'layer_type': 'KAE',
           'model_type': 'AE',
           'order': 3},
 'TRAIN': {'batch_size': 256,
           'epochs': 10,
           'lr': 0.0001,
           'optim_type': 'ADAM',
           'random_seed': 2024,
           'weight_decay': 0.0001}}

{'device': device(type='cuda'),
 'epochs': 10,
 'model': StandardAE(
  (encoder): Sequential(
    (0): DenseLayer(
      (layer): KAELayer(order=3)
    )
    (1): ReLU(inplace=True)
    (2): DenseLayer(
      (layer): KAELayer(order=3)
    )
    (3): ReLU(inplace=True)
  )
  (decoder): Sequential(
    (0): DenseLayer(
      (layer): KAELayer(order=3)
    )
    (1): ReLU(inplace=True)
    (2): DenseLayer(
      (layer): KAELayer(order=3)
    )
    (3): Sigmoid()
  )
),
 'optimizer': Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 0.0001
    maximize: False
    weight_decay: 0.0001
),
 'random_seed': 2024,
 'test_loader': <torch.utils.data.dataloader.DataLoader object at 0x7f187cf11340>,
 'train_loader': <torch.utils.data.dataloader.DataLoader object at 0x7f187cf11040>}

model parameters: 1,639,712

trial 1/10
test loss: 0.016384680382907392, seconds: 326.47234320640564
trial 2/10
test loss: 0.015581317734904588, seconds: 331.4330863952637
trial 3/10
test loss: 0.017017113068141042, seconds: 331.6513261795044
trial 4/10
test loss: 0.015830345428548755, seconds: 327.08414101600647
trial 5/10
test loss: 0.017262230068445204, seconds: 328.4695780277252
trial 6/10
test loss: 0.014666992449201643, seconds: 329.8385400772095
trial 7/10
test loss: 0.014286837540566921, seconds: 337.83651876449585
trial 8/10
test loss: 0.016350188408978283, seconds: 335.5007619857788
trial 9/10
test loss: 0.014879045519046485, seconds: 333.5552954673767
trial 10/10
test loss: 0.0171417145524174, seconds: 331.24688816070557

average/best test loss: 0.01594004651531577 / 0.014286837540566921

```


### KAE (hid=64, p=4), relu/sigmoid, ADAM, lr=0.0001, batch size=256

```
model_config/config6.yaml

updated config:
{'DATA': {'type': 'MNIST'},
 'MODEL': {'hidden_dims': [64],
           'latent_dim': 16,
           'layer_type': 'KAE',
           'model_type': 'AE',
           'order': 4},
 'TRAIN': {'batch_size': 256,
           'epochs': 10,
           'lr': 0.0001,
           'optim_type': 'ADAM',
           'random_seed': 2024,
           'weight_decay': 0.0001}}

{'device': device(type='cuda'),
 'epochs': 10,
 'model': StandardAE(
  (encoder): Sequential(
    (0): DenseLayer(
      (layer): KAELayer(order=4)
    )
    (1): ReLU(inplace=True)
    (2): DenseLayer(
      (layer): KAELayer(order=4)
    )
    (3): ReLU(inplace=True)
  )
  (decoder): Sequential(
    (0): DenseLayer(
      (layer): KAELayer(order=4)
    )
    (1): ReLU(inplace=True)
    (2): DenseLayer(
      (layer): KAELayer(order=4)
    )
    (3): Sigmoid()
  )
),
 'optimizer': Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 0.0001
    maximize: False
    weight_decay: 0.0001
),
 'random_seed': 2024,
 'test_loader': <torch.utils.data.dataloader.DataLoader object at 0x7f195c9638b0>,
 'train_loader': <torch.utils.data.dataloader.DataLoader object at 0x7f195c9638e0>}

model parameters: 512,928

trial 1/10
test loss: 0.019172179815359413, seconds: 147.43125009536743
trial 2/10
test loss: 0.01767512778751552, seconds: 139.19849252700806
trial 3/10
test loss: 0.017656653327867387, seconds: 142.9882152080536
trial 4/10
test loss: 0.019164289673790337, seconds: 138.85816836357117
trial 5/10
test loss: 0.02301151561550796, seconds: 139.53188729286194
trial 6/10
test loss: 0.022694176388904454, seconds: 142.0449194908142
trial 7/10
test loss: 0.01957593164406717, seconds: 147.18947196006775
trial 8/10
test loss: 0.017888742545619608, seconds: 142.23271703720093
trial 9/10
test loss: 0.02021964266896248, seconds: 141.51794171333313
trial 10/10
test loss: 0.022180935135111213, seconds: 143.4875738620758

average/best test loss: 0.019923919460270556 / 0.017656653327867387

```


### KAE (hid=128, p=4), relu/sigmoid, ADAM, lr=0.0001, batch size=256

```
model_config/config6.yaml

updated config:
{'DATA': {'type': 'MNIST'},
 'MODEL': {'hidden_dims': [128],
           'latent_dim': 16,
           'layer_type': 'KAE',
           'model_type': 'AE',
           'order': 4},
 'TRAIN': {'batch_size': 256,
           'epochs': 10,
           'lr': 0.0001,
           'optim_type': 'ADAM',
           'random_seed': 2024,
           'weight_decay': 0.0001}}

{'device': device(type='cuda'),
 'epochs': 10,
 'model': StandardAE(
  (encoder): Sequential(
    (0): DenseLayer(
      (layer): KAELayer(order=4)
    )
    (1): ReLU(inplace=True)
    (2): DenseLayer(
      (layer): KAELayer(order=4)
    )
    (3): ReLU(inplace=True)
  )
  (decoder): Sequential(
    (0): DenseLayer(
      (layer): KAELayer(order=4)
    )
    (1): ReLU(inplace=True)
    (2): DenseLayer(
      (layer): KAELayer(order=4)
    )
    (3): Sigmoid()
  )
),
 'optimizer': Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 0.0001
    maximize: False
    weight_decay: 0.0001
),
 'random_seed': 2024,
 'test_loader': <torch.utils.data.dataloader.DataLoader object at 0x7f195c963040>,
 'train_loader': <torch.utils.data.dataloader.DataLoader object at 0x7f195c963bb0>}

model parameters: 1,025,056

trial 1/10
test loss: 0.01637132160831243, seconds: 255.22766876220703
trial 2/10
test loss: 0.016493089473806323, seconds: 252.3838324546814
trial 3/10
test loss: 0.015356263937428593, seconds: 252.7017102241516
trial 4/10
test loss: 0.015355625795200467, seconds: 256.5977442264557
trial 5/10
test loss: 0.017132615856826305, seconds: 264.0748360157013
trial 6/10
test loss: 0.016231764946132897, seconds: 263.48304653167725
trial 7/10
test loss: 0.015862736152485013, seconds: 261.52497696876526
trial 8/10
test loss: 0.01712557568680495, seconds: 264.40426325798035
trial 9/10
test loss: 0.017479192721657454, seconds: 254.95627284049988
trial 10/10
test loss: 0.01787043015938252, seconds: 256.4294128417969

average/best test loss: 0.0165278616338037 / 0.015355625795200467

```


### KAE (hid=256, p=4), relu/sigmoid, ADAM, lr=0.0001, batch size=256

```
model_config/config6.yaml

updated config:
{'DATA': {'type': 'MNIST'},
 'MODEL': {'hidden_dims': [256],
           'latent_dim': 16,
           'layer_type': 'KAE',
           'model_type': 'AE',
           'order': 4},
 'TRAIN': {'batch_size': 256,
           'epochs': 10,
           'lr': 0.0001,
           'optim_type': 'ADAM',
           'random_seed': 2024,
           'weight_decay': 0.0001}}

{'device': device(type='cuda'),
 'epochs': 10,
 'model': StandardAE(
  (encoder): Sequential(
    (0): DenseLayer(
      (layer): KAELayer(order=4)
    )
    (1): ReLU(inplace=True)
    (2): DenseLayer(
      (layer): KAELayer(order=4)
    )
    (3): ReLU(inplace=True)
  )
  (decoder): Sequential(
    (0): DenseLayer(
      (layer): KAELayer(order=4)
    )
    (1): ReLU(inplace=True)
    (2): DenseLayer(
      (layer): KAELayer(order=4)
    )
    (3): Sigmoid()
  )
),
 'optimizer': Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 0.0001
    maximize: False
    weight_decay: 0.0001
),
 'random_seed': 2024,
 'test_loader': <torch.utils.data.dataloader.DataLoader object at 0x7f195c963100>,
 'train_loader': <torch.utils.data.dataloader.DataLoader object at 0x7f195c963490>}

model parameters: 2,049,312

trial 1/10
test loss: 0.01558066001161933, seconds: 441.3702812194824
trial 2/10
test loss: 0.014235765440389514, seconds: 434.6101531982422
trial 3/10
test loss: 0.015248811431229114, seconds: 428.8143413066864
trial 4/10
test loss: 0.01436346652917564, seconds: 427.17020058631897
trial 5/10
test loss: 0.015836665499955417, seconds: 429.63030791282654
trial 6/10
test loss: 0.015187899000011384, seconds: 429.3934314250946
trial 7/10
test loss: 0.01566753287333995, seconds: 442.68628454208374
trial 8/10
test loss: 0.014760070107877254, seconds: 447.96484541893005
trial 9/10
test loss: 0.014156273938715458, seconds: 444.4883916378021
trial 10/10
test loss: 0.014120391220785677, seconds: 447.82466983795166

average/best test loss: 0.014915753605309872 / 0.014120391220785677

```


### KAE (hid=64, p=5), relu/sigmoid, ADAM, lr=0.0001, batch size=256

```
model_config/config6.yaml

updated config:
{'DATA': {'type': 'MNIST'},
 'MODEL': {'hidden_dims': [64],
           'latent_dim': 16,
           'layer_type': 'KAE',
           'model_type': 'AE',
           'order': 5},
 'TRAIN': {'batch_size': 256,
           'epochs': 10,
           'lr': 0.0001,
           'optim_type': 'ADAM',
           'random_seed': 2024,
           'weight_decay': 0.0001}}

{'device': device(type='cuda'),
 'epochs': 10,
 'model': StandardAE(
  (encoder): Sequential(
    (0): DenseLayer(
      (layer): KAELayer(order=5)
    )
    (1): ReLU(inplace=True)
    (2): DenseLayer(
      (layer): KAELayer(order=5)
    )
    (3): ReLU(inplace=True)
  )
  (decoder): Sequential(
    (0): DenseLayer(
      (layer): KAELayer(order=5)
    )
    (1): ReLU(inplace=True)
    (2): DenseLayer(
      (layer): KAELayer(order=5)
    )
    (3): Sigmoid()
  )
),
 'optimizer': Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 0.0001
    maximize: False
    weight_decay: 0.0001
),
 'random_seed': 2024,
 'test_loader': <torch.utils.data.dataloader.DataLoader object at 0x7f1daaab4fd0>,
 'train_loader': <torch.utils.data.dataloader.DataLoader object at 0x7f1daaab4250>}

model parameters: 615,328

trial 1/10
test loss: 0.01908638181630522, seconds: 145.28206825256348
trial 2/10
test loss: 0.01730156985577196, seconds: 148.1776647567749
trial 3/10
test loss: 0.01892505888827145, seconds: 151.5980896949768
trial 4/10
test loss: 0.019929102598689498, seconds: 159.41308116912842
trial 5/10
test loss: 0.018801589566282927, seconds: 161.51790499687195
trial 6/10
test loss: 0.01736636960413307, seconds: 160.77034425735474
trial 7/10
test loss: 0.017208101460710168, seconds: 161.73465490341187
trial 8/10
test loss: 0.018564854608848692, seconds: 157.03404307365417
trial 9/10
test loss: 0.017425758531317115, seconds: 152.50793719291687
trial 10/10
test loss: 0.01755934504326433, seconds: 152.81868195533752

average/best test loss: 0.018216813197359443 / 0.017208101460710168

```


### KAE (hid=128, p=5), relu/sigmoid, ADAM, lr=0.0001, batch size=256

```
model_config/config6.yaml

updated config:
{'DATA': {'type': 'MNIST'},
 'MODEL': {'hidden_dims': [128],
           'latent_dim': 16,
           'layer_type': 'KAE',
           'model_type': 'AE',
           'order': 5},
 'TRAIN': {'batch_size': 256,
           'epochs': 10,
           'lr': 0.0001,
           'optim_type': 'ADAM',
           'random_seed': 2024,
           'weight_decay': 0.0001}}

{'device': device(type='cuda'),
 'epochs': 10,
 'model': StandardAE(
  (encoder): Sequential(
    (0): DenseLayer(
      (layer): KAELayer(order=5)
    )
    (1): ReLU(inplace=True)
    (2): DenseLayer(
      (layer): KAELayer(order=5)
    )
    (3): ReLU(inplace=True)
  )
  (decoder): Sequential(
    (0): DenseLayer(
      (layer): KAELayer(order=5)
    )
    (1): ReLU(inplace=True)
    (2): DenseLayer(
      (layer): KAELayer(order=5)
    )
    (3): Sigmoid()
  )
),
 'optimizer': Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 0.0001
    maximize: False
    weight_decay: 0.0001
),
 'random_seed': 2024,
 'test_loader': <torch.utils.data.dataloader.DataLoader object at 0x7f195c9638e0>,
 'train_loader': <torch.utils.data.dataloader.DataLoader object at 0x7f195c963280>}

model parameters: 1,229,856

trial 1/10
test loss: 0.015412552421912551, seconds: 329.07769083976746
trial 2/10
test loss: 0.014418959734030068, seconds: 324.79892349243164
trial 3/10
test loss: 0.01620185296051204, seconds: 326.28697967529297
trial 4/10
test loss: 0.014521781401708723, seconds: 320.42595863342285
trial 5/10
test loss: 0.015382755897007883, seconds: 340.09007692337036
trial 6/10
test loss: 0.015597944264300168, seconds: 340.1423282623291
trial 7/10
test loss: 0.015459689102135599, seconds: 341.27795457839966
trial 8/10
test loss: 0.01595883092377335, seconds: 334.1474413871765
trial 9/10
test loss: 0.015460120351053774, seconds: 340.4838047027588
trial 10/10
test loss: 0.016732382122427225, seconds: 327.4674618244171

average/best test loss: 0.015514686917886137 / 0.014418959734030068

```


### KAE (hid=256, p=5), relu/sigmoid, ADAM, lr=0.0001, batch size=256

```
model_config/config6.yaml

updated config:
{'DATA': {'type': 'MNIST'},
 'MODEL': {'hidden_dims': [256],
           'latent_dim': 16,
           'layer_type': 'KAE',
           'model_type': 'AE',
           'order': 5},
 'TRAIN': {'batch_size': 256,
           'epochs': 10,
           'lr': 0.0001,
           'optim_type': 'ADAM',
           'random_seed': 2024,
           'weight_decay': 0.0001}}

{'device': device(type='cuda'),
 'epochs': 10,
 'model': StandardAE(
  (encoder): Sequential(
    (0): DenseLayer(
      (layer): KAELayer(order=5)
    )
    (1): ReLU(inplace=True)
    (2): DenseLayer(
      (layer): KAELayer(order=5)
    )
    (3): ReLU(inplace=True)
  )
  (decoder): Sequential(
    (0): DenseLayer(
      (layer): KAELayer(order=5)
    )
    (1): ReLU(inplace=True)
    (2): DenseLayer(
      (layer): KAELayer(order=5)
    )
    (3): Sigmoid()
  )
),
 'optimizer': Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 0.0001
    maximize: False
    weight_decay: 0.0001
),
 'random_seed': 2024,
 'test_loader': <torch.utils.data.dataloader.DataLoader object at 0x7f195c963a60>,
 'train_loader': <torch.utils.data.dataloader.DataLoader object at 0x7f195c963790>}

model parameters: 2,458,912

trial 1/10
test loss: 0.014011546922847628, seconds: 569.4026217460632
trial 2/10
test loss: 0.014504674705676734, seconds: 571.4599032402039
trial 3/10
test loss: 0.015789117990061642, seconds: 568.842474937439
trial 4/10
test loss: 0.014785659755580128, seconds: 547.6136360168457
trial 5/10
test loss: 0.01338328819256276, seconds: 547.1618297100067
trial 6/10
test loss: 0.013469906221143902, seconds: 546.7046883106232
trial 7/10
test loss: 0.013546474021859467, seconds: 546.1979439258575
trial 8/10
test loss: 0.013565254909917713, seconds: 546.9919567108154
trial 9/10
test loss: 0.014034933387301862, seconds: 546.0972487926483
trial 10/10
test loss: 0.014213305548764765, seconds: 545.7550501823425

average/best test loss: 0.014130416165571657 / 0.01338328819256276

```
