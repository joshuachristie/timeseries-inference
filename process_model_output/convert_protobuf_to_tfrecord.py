"""
Script to convert serialization of the SequenceExample proto definition (in example.proto) to TFRecord.
"""

import tensorflow as tf
import os

directory = "/home/joshua/projects/biofunc-models/data/LSTM/"

for filename in os.listdir(directory):
    if filename.endswith("0"):
        f = open(os.path.join(directory, filename), "rb")
        seq_ex = tf.train.SequenceExample.FromString(f.read())
        with tf.io.TFRecordWriter("{}.tfrecord".format(os.path.join(directory, filename))) as r:
            r.write(seq_ex.SerializeToString())
        f.close()
    else:
        continue

