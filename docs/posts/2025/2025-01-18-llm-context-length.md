---
tags: ["rant", "lm"]
---

# Current standards in Language Model Context Length

So, just reading the [posts on huggingface](https://huggingface.co/posts) today, it seems like
the state-of-the-art context length in large language models is **4 million tokens**.

For models with 400B+ parameters!

I still play around with Microsoft's [Phi 1.5](https://arxiv.org/abs/2309.05463)
which has 0.8B parameters. When quantizing it to 4-bit,
it just fits on my 6 GB graphics card. And i need to fit it there, otherwise it's no fun to talk to
it because it's super slow on CPU.

So, just for comparison, a not even 1B model may fit on your consumer hardware when
quantizing 32 bit floats to 4 bit. With a context length of 2048.

A context length of 4 million tokens is f***ing amazing. That's all of
[Karl Marx's Kapital](https://en.wikipedia.org/wiki/Das_Kapital) and more!
Theoretically, you can process a prompt like

```
Following is a famous three-volume essay about why capitalism sucks.

<insert 'Das Kapital'>

Now, write a representative Python program:
```

On the other hand, what`s the training process for these 4M-context-length models? Did they really
feed **a lot** of 4M token snippets into the model? And if so, what were those snippets?

I mean, i just read the [Phi-4 Tech Report](https://arxiv.org/abs/2412.08905) today about
Microsoft's newest *small* language model which is fed by **a lot** of synthetic data. Basically,
the whole model is nothing new, but the kind of data they fed into is very sophisticated synthetic
stuff that makes it perform on the level of GPT-4o and beyond, at least according to some benchmarks.
The whole paper is a mind-boggling enumeration of sophisticated engineering to design
the pre-, mid- and post-training **data** (indeed, not a single word about model architecture)
and i really wonder what some arbitrary startup has stuffed into their
4M-context-length models? The whole stupid internet?

Well, this is some middle-of-the-night post, and i certainly overdosed myself in the last 48 hours
with neural network topics. Really strange times we are living in right now. People in interviews,
talking about stuff they did in May, as if it was ancient history. And, indeed, it's no
wizardry to proclaim that compute for LLMs will 10-fold within the next year.

I'm certainly interested but also old-fashioned. Never tried GPT myself. Just read a ton of papers
about it. Rather play with Phi-1.5, because it does not require (capitalist's) cloud GPUs to execute.

Unfortunately, it's answers are never sexy:

> Exercise: Write a python program about the ideas of Karl Marx.

```python
# Define the class
class Marx:
    def __init__(self, name, ideology):
        self.name = name
        self.ideology = ideology

# Create an object
marx = Marx("Karl Marx", "Socialism")

# Print the object
print(f"Name: {marx.name}, Ideology: {marx.ideology}")
```
