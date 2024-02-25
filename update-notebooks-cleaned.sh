#!/bin/bash

cp scratchpad/*.py notebooks-cleaned/
cp scratchpad/*.ipynb notebooks-cleaned/
jupyter nbconvert --to notebook --inplace --clear-output notebooks-cleaned/*.ipynb

