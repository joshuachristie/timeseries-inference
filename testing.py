"""
first step is to decide on whether I want to make the data (for train/valid/test) conditional
on having persisted for x generations. If I do, it's definitely a less general approach, as one would
need to train the model specifically for (i.e. conditional upon) sequences that have persisted for x (or more) generations.
So ultimately I'd prefer not to have to do this, but in a way it's a simpler problem. The issue with using all trajectories of allele
frequencies is that the vast majority of these will go extinct in a handful of generations (many in the first generation). This means that
my training set will be heavily biased towards samples that 1. have little to no information, and 2. aren't very interesting.
It's hard to imagine being able to train an accurate model for those cases that do persist for a while if they are <1% of the total samples.

Because I just want to quickly prototype a model to get the framework up, I'll just stick with the version with no conditioning even though
it won't work well (but obviously nothing is going to work well on the first pass while being trained on my laptop).
I'll just stick with datasets small enough to fit in memory and a basic fully-connected nn, focusing on getting the rough outline first.
"""


import pandas as pd
from os import listdir
import re

data_dir = "/home/joshua/projects/metric/data/raw_allele_data/HSE/"

for f in listdir(data_dir):
    # broken because the ncols aren't specified (so it initalises with the first row and then once/if it exceeds that it breaks)
    # for testing might be easier to just branch off metric (and later throw away) and hard code the max number of gens to record raw freqs
    data = pd.read_csv(data_dir + f, nrows=2, header=None)
    data.insert(0,"ps", int(re.search(r'HSE_(.*?)_', f).group(1)))
    data.insert(0,"sc", float(re.search(r'HSE_[0-9]+_(.*?)_', f).group(1)))

data = data.fillna(value=0.0)

print(data.head)
# print(data.iloc[0,2:50])
