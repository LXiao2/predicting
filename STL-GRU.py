import pandas as pd
from keras.layers import Dense, Input
from keras.layers import GRU
from keras.models import Model
from keras.models import load_model
from pandas import DataFrame
from math import ceil
from keras.callbacks import EarlyStopping, ModelCheckpoint
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose

filename = 'DATA.xlsx'
df = pd.read_excel(filename)
df.info()
print(df.isnull().sum())
city = 'city_name1'
period = 7

data = df[city]
seasonal_dec = seasonal_decompose(data, model="additive", period=round(period))
df['trend'] = seasonal_dec.trend
df['seasonal'] = seasonal_dec.seasonal
df['resid'] = seasonal_dec.resid
df.to_excel('data_split.xlsx', index_label=str(period))

split = 0.8
split2 = 0.5
step_in = 5

b = ['trend', 'seasonal', 'resid']

for c in range(len(b)):
    data_all_values = df[b[c]].values.astype('float32')  # df.values
    n_rows = data_all_values.shape[0]

    # %% Normalize
    scaler = MinMaxScaler()
    data_scaled_all_values = scaler.fit_transform(data_all_values)

    # %% split into train validate and test
    data_split_1 = ceil(len(data_scaled_all_values) * split)
    data_train = data_scaled_all_values[:data_split_1, :]
    data_temp = data_scaled_all_values[data_split_1:, :]
    data_split_2 = ceil(len(data_temp) * split2)
    data_valid = data_temp[:data_split_2, :]  #
    data_test = data_temp[data_split_2:, :]  #


    # %% define timer series to supervised learning function
    def series_to_supervised(data1, step, step_out, dropnan=True):
        """
        Frame a time series as a supervised learning dataset.
        Returns:
            Pandas DataFrame of series framed for supervised learning.
        """
        data_value = DataFrame(data1)
        cols = list()
        # input sequence (t-n, ... t-1)
        for i in range(step):
            cols.append(data_value.shift(-i))
        for i in range(step_out):
            cols.append(data_value.shift(-(i + step)))

        # put it all together
        agg = concat(cols, axis=1)
        if dropnan:
            agg.dropna(inplace=True)

        return agg


    data_step_train_values = series_to_supervised(data_train, step_in, 1)
    data_step_valid_values = series_to_supervised(data_valid, step_in, 1)
    data_step_test_values = series_to_supervised(data_test, step_in, 1)

    # split into input and outputs
    n_obs = step_in
    train_X, train_Y = data_step_train_values[:, :n_obs], data_step_train_values[:, n_obs:]
    valid_X, valid_Y = data_step_valid_values[:, :n_obs], data_step_valid_values[:, n_obs:]
    test_X, test_Y = data_step_test_values[:, :n_obs], data_step_test_values[:, n_obs:]

    # Reshape input
    train_X = train_X.reshape((train_X.shape[0], step_in, 1))
    train_Y = train_Y.reshape((train_Y.shape[0], 1, 1))
    valid_X = valid_X.reshape((valid_X.shape[0], step_in, 1))
    valid_Y = valid_Y.reshape((valid_Y.shape[0], 1, 1))
    test_X = test_X.reshape((test_X.shape[0], step_in, 1))
    test_Y = test_Y.reshape((test_Y.shape[0], 1, 1))

    input_layer = Input(shape=(train_X.shape[1], train_X.shape[2]))

    gru_layer = GRU(128)(input_layer)

    output_layer = Dense(1)(gru_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    # model.summary()
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # fit network
    checkpoint = ModelCheckpoint(filepath='model(%s).h5' % (b[c]), monitor='val_loss', verbose=1,
                                 save_best_only=True, save_weights_only=False, mode='auto', period=1)

    callbacks_list = [
        checkpoint,
        EarlyStopping(monitor='val_loss', patience=50,
                      verbose=1)]

    his = model.fit(train_X, train_Y, epochs=10000, batch_size=32,
                    callbacks=callbacks_list,
                    validation_data=(valid_X, valid_Y), verbose=2,
                    shuffle=False)

    # invert scaling
    def invert_scaling(data2):
        data2 = data2.reshape((data2.shape[0], 1))
        inv_data2 = scaler.inverse_transform(data2)
        return DataFrame(inv_data2)


    model = load_model('model(%s).h5' % (b[c]))
    Y_pred = model.predict(test_X)
    inv_Y_pred = invert_scaling(Y_pred)  # invert scaling predicted
    inv_Y_true = invert_scaling(test_Y)  # invert scaling for actual  inv_y= actual test

    result_data = concat([inv_Y_true, inv_Y_pred], axis=1)
    result_data.to_excel('result_data(load_model)(%s).xlsx' % (b[c]))
