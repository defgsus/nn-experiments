
Personal testbed and collection of utilities for torch-based 
neural net experiments.

Folder structure:

- [src/](src/) and [test/](src/) is mostly the library part.
  There is a lot of work on torch Datasets, some of it is
  obsoleted by torch's `DataPipes` library which i do NOT yet use.
- [scripts/](scripts/) contains a couple of different experiments.
  The files are pretty similar in structure and vary in dataset,
  model and training details. Most of it superseded by:
- [experiments/](experiments/) contains new experiments based on
  yaml file descriptions. It's an opinionated wrapper around
  my [Trainer class](src/train/trainer.py) that can also run 
  permutations of parameters and log the results.
- :strawberry: [experiments/logs/](experiments/logs/) contains logbooks of 
  experiments
- [notebooks-cleaned/](notebooks-cleaned/) contains the cleaned (no output)
  version of all the jupyter notebooks used for experimentation. 
  Some of them are useful, some of them are collections of crap from
  middle-of-the-night trial-and-error sessions.

Experiments from the [experiments/](experiments/) folder are run in project root with:

```shell
python exp.py experiments/the-name.yml run
```

Create a github issue if you are interested in playing with it and
need some documentation.
