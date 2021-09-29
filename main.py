from inverse_utils import Model, Data

def main():
    
    data = Data.Dataset()
    Data.convert_binary_to_tfrecord(data.data_directory)
    train_filepaths, valid_filepaths, test_filepaths = data.train_valid_test_split()
    train_dataset, valid_dataset, test_dataset = data.process_datasets(
        train_filepaths, valid_filepaths, test_filepaths)

    model = Model.LSTM()
    model.fit(train_dataset, valid_dataset)

if __name__ == "__main__":
    main()
