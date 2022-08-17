import numpy
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from cond_rnn import ConditionalRecurrent

def svm_regression(train_in, train_out):
    regr = SVR(gamma = 0.01, kernel='rbf', C=10)
    regr.fit(train_in, numpy.ravel(train_out))
    return regr

def rf_regression(train_in, train_out):
    rf = RandomForestRegressor(bootstrap= False, max_depth= 40, max_features = 'auto', min_samples_leaf = 1, min_samples_split = 2, n_estimators = 600)
    rf.fit(train_in, numpy.ravel(train_out))
    return rf

def svm_regression_delta(train_in, train_out):
    regr = SVR(gamma = 0.01, kernel='rbf', C=1)
    regr.fit(train_in, numpy.ravel(train_out))
    return regr

import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.regularizers import l2

def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def dnn_velocity(train_in, train_out, test_in, test_out):
    learningRate = 0.001
    model = keras.Sequential([
        keras.layers.Dense(len(train_in[0]), activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(.2),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(1)
    ])
    callback = keras.callbacks.EarlyStopping(monitor='loss', patience=10)

    model.compile(
        loss='mean_absolute_error',
        #loss='mean_squared_error',   # This one is more sensitive to outliers
        optimizer=tf.keras.optimizers.Adam(learningRate),
        metrics = ['mse', coeff_determination]
    )
    model.build(input_shape = train_in.shape)

    model.summary()

    history = model.fit(train_in,
                        train_out,
                        validation_data=(test_in, test_out),
                        verbose=False,
                        epochs=500,
                        callbacks = [callback]
                        #batch_size=32,
    )
    return model, history

def dnn_model(train_in, train_out, test_in, test_out) :

    learningRate = 0.001
    model = keras.Sequential([
        # normalizer,
        keras.layers.Dense(len(train_in[0]), activation='relu'),
        # keras.layers.Dense(16, activation='relu'),
        keras.layers.Dropout(.2),
        keras.layers.Dense(32, activation='relu'),
        # keras.layers.Dense(4, activation='relu'),
        keras.layers.Dense(1)
    ])

    callback = keras.callbacks.EarlyStopping(monitor='loss', patience=10)

    model.compile(
        loss='mean_absolute_error',
        #loss='mean_squared_error',   # This one is more sensitive to outliers
        optimizer=tf.keras.optimizers.Adam(learningRate),
        metrics = ['mse', coeff_determination]
    )

    model.build(input_shape = train_in.shape)

    model.summary()

    history = model.fit(train_in,
                        train_out,
                        validation_data=(test_in, test_out),
                        verbose=False,
                        epochs=50,
                        callbacks = [callback]
    )

    return model, history

def lstm_velocity(train_generator, test_generator, timesteps, n_features):
    # CREATING THE MODEL
    # learningRate = 0.0001
    callback = keras.callbacks.EarlyStopping(monitor='loss', patience=10)
    lstm_model = keras.models.Sequential()

    # basic lstm
    # lstm_model.add(keras.layers.LSTM(len(INPUT_VARIABLES), input_shape=(None, n_features), dropout=0.1, recurrent_dropout=0.5, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01)))

    # stacking lstm 
    lstm_model.add(keras.layers.LSTM(400, 
                                    # dropout=0.1, recurrent_dropout=0.5, 
                                    kernel_regularizer=l2(0.0001),
                                    recurrent_regularizer=l2(0.00001),
                                    return_sequences=True,
                                    # stateful = True,
                                    # batch_input_shape = (batch_size, length, n_features),
                                    input_shape=(timesteps, n_features)))
    lstm_model.add(keras.layers.Dense(20, activation='tanh'))
    lstm_model.add(keras.layers.LSTM(300, activation='tanh'))
    lstm_model.add(keras.layers.Dense(20, activation='tanh'))
    lstm_model.add(keras.layers.Dropout(0.25))

    lstm_model.add(keras.layers.Dense(1))
    lstm_model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mean_absolute_error', metrics = ['mse', coeff_determination])

    lstm_model.build()
    lstm_model.summary()
    # Fit the model
    history = lstm_model.fit(train_generator, epochs=100, steps_per_epoch=100, verbose=False, validation_data=test_generator, callbacks=[callback])

    return lstm_model, history

def lstm_lat(train_generator, test_generator, timesteps, n_features):
    callback = keras.callbacks.EarlyStopping(monitor='loss', patience=10)
    lstm_model = keras.models.Sequential()

    # basic lstm
    # lstm_model.add(keras.layers.LSTM(len(INPUT_VARIABLES), input_shape=(None, n_features), dropout=0.1, recurrent_dropout=0.5, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01)))

    # stacking lstm 
    lstm_model.add(keras.layers.LSTM(200, 
                                    kernel_regularizer=l2(0.0001),
                                    recurrent_regularizer=l2(0.00001),
                                    input_shape=(timesteps, n_features)))
    lstm_model.add(keras.layers.Dropout(0.25))

    lstm_model.add(keras.layers.Dense(1))
    lstm_model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mean_absolute_error', metrics = ['mse', coeff_determination])

    lstm_model.build()
    lstm_model.summary()
    # Fit the model
    history = lstm_model.fit(train_generator, epochs=10, steps_per_epoch=100, verbose=False, validation_data=test_generator, callbacks=[callback])
    return lstm_model, history

def condrnn_vs(train_in, train_c, train_out, test_in, test_out, test_c):
    # DEFINING CALLBACKS
    callback = keras.callbacks.EarlyStopping(monitor='loss', patience=5)

    # CREATING THE MODEL
    cond_rnn_model = keras.models.Sequential(layers=[ConditionalRecurrent(keras.layers.LSTM(400, 
                                                                        kernel_regularizer=l2(0.0001), 
                                                                        recurrent_regularizer=l2(0.00001), 
                                                                        return_sequences = True
                                                                        )),
                                                    keras.layers.Dense(units=20, activation='tanh'),
                                                    keras.layers.LSTM(300, activation = 'tanh'),
                                                    keras.layers.Dense(units=20, activation='tanh'),
                                                    keras.layers.Dense(units=1, activation='linear')])
    cond_rnn_model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mean_absolute_error', metrics = ['mse', coeff_determination])
    history = cond_rnn_model.fit(x = [train_in, train_c], y = train_out, epochs=40, verbose=False, validation_data=([test_in, test_c], test_out), callbacks=[callback])
    return cond_rnn_model, history