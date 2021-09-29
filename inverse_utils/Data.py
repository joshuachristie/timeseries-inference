import random
import os
import tensorflow as tf

class Dataset():

    def __init__(self, data_directory="./data", train_prop=0.6, valid_prop=0.2, seed=42,
                 tf_shuffle=True):
        self.data_directory = data_directory
        self.train_prop = train_prop
        self.valid_prop = valid_prop
        self.seed = seed
        self.tf_shuffle = tf_shuffle
        self.number_train_files = None
        self.number_valid_files = None
        self.number_test_files = None
    
    def train_valid_test_split(self):
        
        random.seed(self.seed)

        list_files = [os.path.join(self.data_directory, f) for f in os.listdir(self.data_directory) if
                      os.path.isfile(os.path.join(self.data_directory, f)) and f.endswith('.tfrecord')]
        
        random.shuffle(list_files)
        
        self.number_train_files = round(len(list_files) * self.train_prop)
        self.number_valid_files = round(len(list_files) * self.valid_prop)
        self.number_test_files = len(list_files) - self.number_train_files - self.number_valid_files

        train_filepaths = [f for f in list_files[0: self.number_train_files]]
        valid_filepaths = [f for f in list_files[self.number_train_files: self.number_train_files + self.number_valid_files]]
        test_filepaths = [f for f in list_files[-self.number_test_files:]]
                                                        
        return train_filepaths, valid_filepaths, test_filepaths
       
    def __filepaths_to_dataset(self, filepaths):

        filepath_dataset = tf.data.TFRecordDataset.from_tensor_slices(filepaths)
        dataset = filepath_dataset.interleave(
            lambda x: tf.data.TFRecordDataset(x),
            cycle_length=filepath_dataset.cardinality(),
            block_length=1,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            deterministic=not self.tf_shuffle)

        return dataset

    def process_datasets(self, train_filepaths, valid_filepaths, test_filepaths):
        
        batch_size = 1024
        num_trait_data = 1000  # need to clean this up, currently this is hardcoded both here and in the preprocess function
        
        train_dataset = self.__filepaths_to_dataset(train_filepaths)
        valid_dataset = self.__filepaths_to_dataset(valid_filepaths)
        test_dataset = self.__filepaths_to_dataset(test_filepaths)
        
        train_dataset = train_dataset.interleave(
            preprocess_and_split_into_tuples,
            cycle_length=self.number_train_files,
            block_length=1,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            deterministic=False)

        valid_dataset = valid_dataset.interleave(
            preprocess_and_split_into_tuples,
            cycle_length=self.number_valid_files,
            block_length=1,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            deterministic=False)

        test_dataset = test_dataset.interleave(
            preprocess_and_split_into_tuples,
            cycle_length=self.number_test_files,
            block_length=1,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            deterministic=False)

        train_dataset = train_dataset.shuffle(buffer_size=num_trait_data*self.number_train_files, reshuffle_each_iteration=True).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        
        valid_dataset = valid_dataset.shuffle(buffer_size=num_trait_data*self.number_valid_files, reshuffle_each_iteration=True).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        
        test_dataset = test_dataset.shuffle(buffer_size=num_trait_data*self.number_test_files, reshuffle_each_iteration=True).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        return train_dataset, valid_dataset, test_dataset
    

    # note that I still need to handle the different implementation of the test set (it currently predicts every example where train does fewer)

def preprocess_and_split_into_tuples(tfrecord):
    
    context_features = {"population_size" : tf.io.FixedLenFeature((), tf.int64),
                        "selection_coefficient" : tf.io.FixedLenFeature((), tf.float32)}
    sequence_features = {"raw_trait_frequencies" : tf.io.RaggedFeature(tf.float32)}

    pop_size_min = 50
    pop_size_max = 500
    sc_min = 0.0
    sc_max = 1.0
    # the minimum number of gens actually survived by the trait is min_survival - 1 
    # (-2 because the starting and final freq are always recorded; +1 because I filter using > not >=)
    min_survival = 0
    max_time_steps = 60 # truncate trait frequencies at max_time_steps to avoid OOM issues
    num_trait_data = 1000

    # parse sequence example and normalise
    example = tf.io.parse_sequence_example(tfrecord, context_features=context_features, sequence_features=sequence_features)
    trait_frequencies = example[1]["raw_trait_frequencies"]
    pop_size_norm = (tf.cast(example[0]["population_size"], dtype=tf.float32) - pop_size_min) / (pop_size_max - pop_size_min)
    sc_norm = (tf.cast(example[0]["selection_coefficient"], dtype=tf.float32) - sc_min) / (sc_max - sc_min)
    # filter dataset to only include num_trait_data trials in which the trait survives at least min_survival
    trait_frequencies = tf.gather(trait_frequencies, tf.where(trait_frequencies.row_lengths() > min_survival), axis=0)

    trait_frequencies = tf.gather(trait_frequencies, tf.random.uniform(shape=[num_trait_data], minval=0,
                                                                      maxval=trait_frequencies.nrows(),
                                                                      dtype=tf.int32))    
    trait_frequencies = trait_frequencies[:, :, 0:max_time_steps] 
    # cast labels into a list of tensors matching the features
    label_tensors = tf.reshape(tf.stack((tf.repeat(pop_size_norm, num_trait_data), tf.repeat(sc_norm, num_trait_data)), axis=1), 
                               shape=(num_trait_data, 2))   
    # convert trait_frequencies to dense tensor (with mask value of -1)
    trait_frequencies = tf.squeeze(trait_frequencies, axis=1)
    trait_frequencies = trait_frequencies.to_tensor(default_value = -1., 
                                                    shape=(num_trait_data, max_time_steps))    
    # add final dimension (keras requires an input shape of (batch_size, timesteps, features))
    trait_frequencies = tf.expand_dims(trait_frequencies, axis=2)

    return tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(trait_frequencies), tf.data.Dataset.from_tensor_slices(label_tensors)))


def convert_binary_to_tfrecord(data_directory):
    for filename in os.listdir(data_directory):
        if filename.endswith("0"):
            full_filename = os.path.join(data_directory, filename)
            with open(full_filename, "rb") as f:
                seq_ex = tf.train.SequenceExample.FromString(f.read())
            with tf.io.TFRecordWriter(f'{full_filename}.tfrecord') as r:
                r.write(seq_ex.SerializeToString())
            os.remove(full_filename)

        else:
            continue
    return
