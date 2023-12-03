#!/bin/bash

cp scratchpad/*.ipynb notebooks-cleaned/
jupyter nbconvert --to notebook --inplace --clear-output notebooks-cleaned/*.ipynb

