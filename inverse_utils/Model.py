import tensorflow as tf
import tensorflow_addons as tfa
from datetime import datetime

class LSTM():

    def __init__(self, units=75, recurrent_dropout=0.25, dropout=0.2, dense_units=30,
                 learning_rate=0.01, verbose=False, train=True):
        
        self.units = units
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.dense_units = dense_units
        self.lr = learning_rate
        self.callbacks = None
        self.metrics = [tf.keras.metrics.MeanAbsoluteError()]
        self.optimizer = tf.keras.optimizers.Adam(self.lr)
        self.loss = tf.keras.losses.MeanSquaredError()
        
        self.model = self.build_model(train, verbose)

            # add code to load best model weights

    def build_model(self, train, verbose):
        
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Masking(mask_value=-1., input_shape=(None, 1)))
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.RNN(
            tfa.rnn.LayerNormLSTMCell(self.units, recurrent_dropout=self.recurrent_dropout), return_sequences=True)))
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.RNN(
            tfa.rnn.LayerNormLSTMCell(self.units, recurrent_dropout=self.recurrent_dropout), return_sequences=False)))
        model.add(tf.keras.layers.Dropout(self.dropout))
        model.add(tf.keras.layers.Dense(self.dense_units, activation='elu', kernel_initializer='he_normal'))
        model.add(tf.keras.layers.Dropout(self.dropout))
        model.add(tf.keras.layers.Dense(2))
        
        def scheduler(epoch):
            if epoch < 4:
                return self.lr
            else:
                return self.lr * tf.math.exp(-0.1)

        scheduler_cb = tf.keras.callbacks.LearningRateScheduler(scheduler)
        earlystopping_cb = tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)

        checkpoints = "checkpoints/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        checkpoint_filepath = checkpoints + "checkpoint_layernorm_{epoch:02d}"

        model_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=False,
            save_freq='epoch')

        self.callbacks = [model_checkpoint_cb, scheduler_cb, earlystopping_cb]
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        
        if not train:
            model.load_weights('best_model_june11')
            
        if verbose:
            model.summary()
        
        return model


    def fit(self, train_dataset, valid_dataset):
        history = self.model.fit(train_dataset, epochs=50, validation_data=valid_dataset,
                                 callbacks=self.callbacks)
        return history




