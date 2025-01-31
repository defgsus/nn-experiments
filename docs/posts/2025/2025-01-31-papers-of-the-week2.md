---
title: Papers of the week
tags: ["pow"]
---

Three papers about language models i had fun reading.


## How Large Language Models (LLMs) Extrapolate: From Guided Missiles to Guided Prompts

*Xuenan Cao*

[arxiv:2501.10361](https://arxiv.org/pdf/2501.10361)

I don't really know what the message of this paper is supposed to be, except that, what
today's next-token-prediction language models do, is extrapolation. So nobody should wonder
when they give information that is essentially wrong. It *could* have been true in a parallel
universe. 

It's more a historical essay about Norbert Wiener, Andrey Kolmogorov and the hot and cold war
and contains many nice figures of speech, like, 
*A transmission apparatus tosses out a word like a jet ejecting a bomb*.


## On the Dangers of Stochastic Parrots: Can Language Models Be Too Big?

*Emily M. Bender, Timnit Gebru, Angelina McMillan-Major, Shmargaret Shmitchell*

[DOI:10.1145/3442188.3445922](https://dl.acm.org/doi/10.1145/3442188.3445922)

One of the papers, cited in the above *Extrapolation* essay. At the recent pace in LLM engineering,
a paper from 2021 is already *ancient*. Nevertheless, it's an interesting read with many valid
critiques that are still not really addressed. E.g., the idea that diversity and unbiased 
comprehension arises when feeding in the whole of the internet (excluding texts with *bad* words). 


## Uncovering Deceptive Tendencies in Language Models: A Simulated Company AI Assistant

*Olli Järviniemi, Evan Hubinger*

[arxiv:2405.0157](https://arxiv.org/pdf/2405.0157)

What if, an LLM works for a company and colleagues address it with all sorts of
(maybe not entirely ethical) tasks during the day. 
It also has access to company chat and through that, is informed 
that an evaluator will soon appear to see if the model abides AI regulations. 
Will it talk honestly to the evaluator?

It's an interesting story the authors have created, a bit like a text adventure, 
and i'm honestly impressed what amount of text an LLM like Claude 3 Opus can handle. 
