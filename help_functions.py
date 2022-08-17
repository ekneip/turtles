import pickle
import os
import time
import matplotlib.pyplot as plt
import numpy
import pandas
from scipy.signal import savgol_filter
from pykalman import KalmanFilter
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from data_preparation import prepare_data
from settings import INPUT_VARIABLES, MODEL, OUTPUT_VAR, VERBOSE
from models import condrnn_vs, dnn_model, dnn_velocity, lstm_lat, svm_regression, svm_regression_delta, rf_regression, lstm_velocity
from keras.preprocessing.sequence import TimeseriesGenerator

from datetime import datetime
import matplotlib.dates as mdates
if not os.path.exists('svm_models/thesis_models/'):
        os.makedirs('svm_models/thesis_models/')
def plot_predictions(test_out, pred_out, j):
    fig, ax = plt.subplots()
    ax.scatter(test_out, pred_out)

    lims = [
        numpy.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        numpy.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]

    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    plt.xlabel('Expected values')
    plt.ylabel('Predictions')
    if not os.path.exists('images_thesis/'+MODEL+'/'):
        os.makedirs('images_thesis/'+MODEL+'/')
    plt.savefig('images_thesis/'+MODEL+'/'+str(j) + '_' +OUTPUT_VAR+'_predictions.png')
    plt.clf()
    # plt.show()

def plot_loss(history, j):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.xlabel('Epoch')
  plt.ylabel('Error')
  plt.legend()
  plt.grid(True)
  if not os.path.exists('images_thesis/'+MODEL+'/'):
    os.makedirs('images_thesis/'+MODEL+'/')
  plt.savefig('images_thesis/'+MODEL+'/'+str(j) + '_' +OUTPUT_VAR+'_loss.png')
  plt.clf()
  # plt.show()

def plot_r2(history, j):
  plt.plot(history.history['coeff_determination'], label='r2')
  plt.plot(history.history['val_coeff_determination'], label='val_r2')
  plt.xlabel('Epoch')
  plt.ylabel('R²')
  plt.legend()
  plt.grid(True)
  if not os.path.exists('images_thesis/'+MODEL+'/'):
    os.makedirs('images_thesis/'+MODEL+'/')
  plt.savefig('images_thesis/'+MODEL+'/'+str(j) + '_' +OUTPUT_VAR+'_r2.png')
  plt.clf()
  # plt.show()

def print_latitudes(model, turtle_in, turtle_date, turtle_out, test_turtle, verbose, j):
    for i in range(len(turtle_date)):
        new_date = datetime.fromisoformat(turtle_date[i])
        turtle_date[i] = new_date

    y_pred = model.predict(turtle_in)
    # kf = KalmanFilter(initial_state_mean=36, n_dim_obs=1)
    # kf = kf.em(y_pred)
    # y_filtered_kf = kf.smooth(y_pred)[0]
    acc = r2_score(turtle_out, y_pred)
    print(test_turtle+ ' - r² :: ', acc)
    print(test_turtle+ ' - loss MAE :: ', mean_absolute_error(turtle_out, y_pred))
    if MODEL == 'svm-savgol' :
        if OUTPUT_VAR == 'lat':
            y_filtered = savgol_filter(numpy.ravel(y_pred), 3, 1)
        else:
            y_filtered = savgol_filter(numpy.ravel(y_pred), 11, 6)
            # y_filtered = savgol_filter(numpy.ravel(y_pred), 3, 1)
        filter_acc = r2_score(turtle_out, y_filtered)
        print(test_turtle+ ' - savgol r² :: ', filter_acc)
        print(test_turtle+ ' - savgol loss MAE :: ', mean_absolute_error(turtle_out, y_filtered))
    # kf_filter_acc = r2_score(turtle_out, y_filtered_kf)
    # print('PRINT - Kalman r² :: ', kf_filter_acc)
    # print('PRINT - Kalman loss MAE :: ', mean_absolute_error(turtle_out, y_filtered_kf))
    if verbose == 2:
        if OUTPUT_VAR == 'lat':
            outputvar = 'latitudes'
        elif OUTPUT_VAR == 'lon':
            outputvar = 'longitudes'
        elif OUTPUT_VAR == 'delta_lat':
            outputvar = 'latitude changes'
        elif OUTPUT_VAR == 'delta_lon':
            outputvar = 'longitude changes'
        elif OUTPUT_VAR == 'vs':
            outputvar = 'velocity'
        else:
            outputvar = 'latitudes' 
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.plot(turtle_date, turtle_out, label='actual ' + outputvar)
        if MODEL == 'svm-savgol' :
            plt.plot(turtle_date, y_filtered, label= 'predicted '+outputvar, color = 'chocolate')
        else :
            plt.plot(turtle_date, y_pred, label= 'predicted '+ outputvar, color = 'chocolate')
        # plt.plot(turtle_date, y_filtered_kf, label= 'Kalman filtered predictions', color = 'purple')
        # plt.plot(turtle_date, y_filtered, label= 'Savitzky-Golay filtered predictions', color = 'chocolate')
        plt.legend()
        plt.ylabel(OUTPUT_VAR)
        plt.gcf().autofmt_xdate()
        if not os.path.exists('images_thesis/'+MODEL+'/'):
            os.makedirs('images_thesis/'+MODEL+'/')
        plt.savefig('images_thesis/'+MODEL+'/'+str(j) + '_' +OUTPUT_VAR + '_' + test_turtle+'.png')
        plt.clf()
        # plt.show()

def print_latitudes_lstm(model, turtle_generator, turtle_date, turtle_out, test_turtle, verbose, j):
    for i in range(len(turtle_date)):
        new_date = datetime.fromisoformat(turtle_date[i])
        turtle_date[i] = new_date

    y_pred = model.predict(turtle_generator)
    acc = r2_score(turtle_out, y_pred)
    print(test_turtle+ ' - r² :: ', acc)
    print(test_turtle+ ' - loss MAE :: ', mean_absolute_error(turtle_out, y_pred))
    if verbose == 2:
        if OUTPUT_VAR == 'lat':
            outputvar = 'latitudes'
        elif OUTPUT_VAR == 'lon':
            outputvar = 'longitudes'
        elif OUTPUT_VAR == 'delta_lat':
            outputvar = 'latitude changes'
        elif OUTPUT_VAR == 'delta_lon':
            outputvar = 'longitude changes'
        elif OUTPUT_VAR == 'vs':
            outputvar = 'velocity'
        else:
            outputvar = 'latitudes' 
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.plot(turtle_date, turtle_out, label='actual ' + outputvar)
        plt.plot(turtle_date, y_pred, label= 'predicted '+ outputvar, color = 'chocolate')
        plt.legend()
        plt.gcf().autofmt_xdate()
        if not os.path.exists('images_thesis/'+MODEL+'/'):
            os.makedirs('images_thesis/'+MODEL+'/')
        plt.savefig('images_thesis/'+MODEL+'/'+str(j) + '_' +OUTPUT_VAR + '_' + test_turtle+'.png')
        plt.clf()

def run_model(vars, j, verbose = 0, lag = None):
    train_data, test_data, poundnet_turtle, large_captive, medium_captive, small_captive, wild_data = prepare_data(vars, lag)
    train_in = train_data[:,:-3]
    train_out = train_data[:, -3]

    poundnet_in = poundnet_turtle[:,:-3]
    poundnet_out = poundnet_turtle[:, -3]
    poundnet_date = poundnet_turtle[:, -2]
    
    large_in = large_captive[:,:-3]
    large_out = large_captive[:, -3]
    large_date = large_captive[:, -2]

    medium_in = medium_captive[:,:-3]
    medium_out = medium_captive[:, -3]
    medium_date = medium_captive[:, -2]

    small_in = small_captive[:,:-3]
    small_out = small_captive[:, -3]
    small_date = small_captive[:, -2]

    test_in = test_data[:,:-3]
    test_out = test_data[:, -3]
    
    wild_in = wild_data[:,:-3]
    wild_out = wild_data[:,-3]

    train_in = numpy.asarray(train_in).astype('float32')
    train_out = numpy.asarray(train_out).astype('float32')
    test_in = numpy.asarray(test_in).astype('float32')
    test_out = numpy.asarray(test_out).astype('float32')
    large_in = numpy.asarray(large_in).astype('float32')
    large_out = numpy.asarray(large_out).astype('float32')
    poundnet_in = numpy.asarray(poundnet_in).astype('float32')
    poundnet_out = numpy.asarray(poundnet_out).astype('float32')
    medium_in = numpy.asarray(medium_in).astype('float32')
    medium_out = numpy.asarray(medium_out).astype('float32')
    small_in = numpy.asarray(small_in).astype('float32')
    small_out = numpy.asarray(small_out).astype('float32')
    wild_in = numpy.asarray(wild_in).astype('float32')
    wild_out = numpy.asarray(wild_out).astype('float32')

    if MODEL == 'lstm':
        predicts_out = []

        if OUTPUT_VAR == 'lat':
            scaler = StandardScaler()
            train_out = scaler.fit_transform(train_out.reshape(-1, 1))
            train_in = numpy.column_stack((train_in, train_out))
            scaler = StandardScaler()
            test_out = scaler.fit_transform(test_out.reshape(-1, 1))
            test_in = numpy.column_stack((test_in, test_out))
            scaler = StandardScaler()
            poundnet_out = scaler.fit_transform(poundnet_out.reshape(-1, 1))
            poundnet_in = numpy.column_stack((poundnet_in, poundnet_out))
            scaler = StandardScaler()
            large_out = scaler.fit_transform(large_out.reshape(-1, 1))
            large_in = numpy.column_stack((large_in, large_out))
            scaler = StandardScaler()
            medium_out = scaler.fit_transform(medium_out.reshape(-1, 1))
            medium_in = numpy.column_stack((medium_in, medium_out))
            scaler = StandardScaler()
            small_out = scaler.fit_transform(small_out.reshape(-1, 1))
            small_in = numpy.column_stack((small_in, small_out))
            scaler = StandardScaler()
            wild_out = scaler.fit_transform(wild_out.reshape(-1,1))
            wild_in = numpy.column_stack((wild_in, wild_out))
        else:
            train_in = numpy.column_stack((train_in, train_out))
            test_in = numpy.column_stack((test_in, test_out))
            poundnet_in = numpy.column_stack((poundnet_in, poundnet_out))
            large_in = numpy.column_stack((large_in, large_out))
            medium_in = numpy.column_stack((medium_in, medium_out))
            small_in = numpy.column_stack((small_in, small_out))
            wild_in = numpy.column_stack((wild_in, wild_out))
        
        length = 5
        batch_size = 363
        timesteps = len(train_in)
        n_features = len(INPUT_VARIABLES)+1

        if OUTPUT_VAR == 'vs' :
            train_generator = TimeseriesGenerator(train_in, train_out,
                                            length=length, batch_size=batch_size)
            test_generator = TimeseriesGenerator(test_in, test_out,
                                            length=length, batch_size=batch_size)
            medium_generator = TimeseriesGenerator(medium_in, medium_out,
                                            length=length, batch_size=batch_size)
            regr_model, history = lstm_velocity(train_generator, test_generator, length, n_features)
            predicts_out.append(regr_model.predict(medium_generator))

            for day in range(1,5):
                train_out_day = numpy.roll(train_out, -day, axis = 0)
                train_in_day = train_in[:-day,:]
                train_out_day = train_out_day[:-day]
                test_out_day = numpy.roll(test_out, -day, axis = 0)
                test_in_day = test_in[:-day,:]
                test_out_day = test_out_day[:-day]
                medium_out_day = numpy.roll(medium_out, -day, axis = 0)
                medium_in_day = medium_in[:-day,:]
                medium_out_day = medium_out_day[:-day]   
                
                # GENERATING THE TIMESERIES    
                
                train_generator = TimeseriesGenerator(train_in_day, train_out_day,
                                                length=length, batch_size=batch_size)
                test_generator = TimeseriesGenerator(test_in_day, test_out_day,
                                                length=length, batch_size=batch_size)
                medium_generator = TimeseriesGenerator(medium_in_day, medium_out_day,
                                                length=length, batch_size=batch_size)
                regr_model, history = lstm_velocity(train_generator, test_generator, length, n_features)
                predicts_out.append(regr_model.predict(medium_generator))

            fig, ax = plt.subplots()
            for i in range(len(medium_date)):
                new_date = datetime.fromisoformat(medium_date[i])
                medium_date[i] = new_date
            ax.plot(medium_date[length:], medium_out[length:], color='tab:blue')
            colors = ['maroon', 'firebrick', 'chocolate', 'darkorange', 'orange', 'gold']
            for i in range(5):
                predict_out = predicts_out[i]
                points_i = medium_date[length+i:length+i+1+len(predict_out)]
                ax.plot(points_i, predict_out, color = colors[i], label = 'day ' + str(i+1) + ' predictions')

            legend = ax.legend()
            plt.ylabel('vs')
            if not os.path.exists('images_thesis/'+MODEL+'/'):
                os.makedirs('images_thesis/'+MODEL+'/')
            plt.savefig('images_thesis/'+MODEL+'/'+str(j) + '_' +OUTPUT_VAR + '_daysPredictions.png')

        if OUTPUT_VAR == 'vs':
            length = 5
        else:
            length = 1
        batch_size = 363
        # GENERATING THE TIMESERIES    
        n_features = len(INPUT_VARIABLES)+1
        
        train_generator = TimeseriesGenerator(train_in, train_out,
                                        length=length, batch_size=batch_size)
        test_generator = TimeseriesGenerator(test_in, test_out,
                                        length=length, batch_size=batch_size)
        poundnet_generator = TimeseriesGenerator(poundnet_in, poundnet_out,
                                        length=length, batch_size=batch_size)
        large_generator = TimeseriesGenerator(large_in, large_out,
                                        length=length, batch_size=batch_size)
        medium_generator = TimeseriesGenerator(medium_in, medium_out,
                                        length=length, batch_size=batch_size)
        small_generator = TimeseriesGenerator(small_in, small_out,
                                        length=length, batch_size=batch_size)
        wild_generator = TimeseriesGenerator(wild_in, wild_out,
                                        length=length, batch_size=batch_size)

        if OUTPUT_VAR == 'vs':
            regr_model, history = lstm_velocity(train_generator, test_generator, length, n_features)
        else : 
            regr_model, history = lstm_lat(train_generator, test_generator, length, n_features)
        plot_loss(history, j)
        plot_r2(history, j)
        y_pred = regr_model.predict(test_generator)
        plot_predictions(test_out[length:], y_pred, j)
        y_wild_pred = regr_model.predict(wild_generator)
        acc = r2_score(test_out[length:], y_pred)
        print(MODEL + '_'+ str(j)+' - r² :: ', acc)
        print(MODEL + '_'+ str(j)+' - loss MAE :: ', mean_absolute_error(test_out[length:], y_pred))
        
        acc_wild = r2_score(wild_out[length:], y_wild_pred)
        print(MODEL + '_WILD_'+ str(j)+' - r² :: ', acc_wild)
        print(MODEL + '_WILD_'+ str(j)+' - loss MAE :: ', mean_absolute_error(wild_out[length:], y_wild_pred))

        if VERBOSE != 0:
            print_latitudes_lstm(regr_model, poundnet_generator, poundnet_date[length:], poundnet_out[length:], 'poundnet', verbose,j)
            print_latitudes_lstm(regr_model, large_generator, large_date[length:], large_out[length:], 'large', verbose,j)
            print_latitudes_lstm(regr_model, medium_generator, medium_date[length:], medium_out[length:],'medium', verbose,j)
            print_latitudes_lstm(regr_model, small_generator, small_date[length:], small_out[length:], 'small', verbose,j)
        return acc

    else : 
        tic = time.time()
        if MODEL == 'dnn':
            if OUTPUT_VAR == 'lat':
                regr_model, history = dnn_model(train_in, train_out, test_in, test_out)
                plot_loss(history, j)
                plot_r2(history, j)
            else :
                regr_model, history = dnn_velocity(train_in, train_out, test_in, test_out)
                plot_loss(history, j)
                plot_r2(history, j)
        elif MODEL == 'svm-savgol':
            if os.path.isfile('svm_models/thesis_models/'+MODEL +'_'+ OUTPUT_VAR+'_'+str(j) +'_model.pkl'):
                regr_model = pickle.load(open('svm_models/thesis_models/'+MODEL + '_'+ OUTPUT_VAR+'_'+str(j) +'_model.pkl', 'rb'))
            else:    
                regr_model = svm_regression(train_in, train_out)
                pickle.dump(regr_model, open('svm_models/thesis_models/'+MODEL +'_'+ OUTPUT_VAR+'_'+str(j) +'_model.pkl', 'wb'))
        elif MODEL == 'rf':
            regr_model = rf_regression(train_in, train_out)
        elif MODEL == 'svm':
            if os.path.isfile('svm_models/thesis_models/'+MODEL + '_' + str(lag)+'_'+ OUTPUT_VAR+'_'+str(j) +'_model.pkl'):
                regr_model = pickle.load(open('svm_models/thesis_models/'+MODEL + '_' + str(lag)+'_'+ OUTPUT_VAR+'_'+str(j) +'_model.pkl', 'rb'))
            else:    
                regr_model = svm_regression_delta(train_in, train_out)
                pickle.dump(regr_model, open('svm_models/thesis_models/'+MODEL + '_' + str(lag)+'_'+ OUTPUT_VAR+'_'+str(j) +'_model.pkl', 'wb'))
        elif MODEL == 'lstm':
            regr_model, history = None, None
                

        toc = time.time()
        print('TIME USED ' + str(j)+' ',(toc-tic))

        y_pred = regr_model.predict(test_in)
        plot_predictions(test_out, y_pred, j)
        y_wild_pred = regr_model.predict(wild_in)

        if MODEL == 'svm-savgol':
            if OUTPUT_VAR == 'lat':
                y_filtered = savgol_filter(numpy.ravel(y_pred), 3, 1)
                y_wild_filtered = savgol_filter(numpy.ravel(y_wild_pred), 3, 1)
            else:
                y_filtered = savgol_filter(numpy.ravel(y_pred), 11, 6)
                y_wild_filtered = savgol_filter(numpy.ravel(y_wild_pred), 11, 6)
            filter_acc = r2_score(test_out, y_filtered)

            print(MODEL + '_'+ str(j) +' - savgol r² :: ', filter_acc)
            print(MODEL + '_'+ str(j)+' - savgol loss MAE :: ', mean_absolute_error(test_out, y_filtered))

            filter_acc_wild = r2_score(wild_out, y_wild_filtered)
            print(MODEL + '_WILD_'+ str(j)+' - savgol r² :: ', filter_acc_wild)
            print(MODEL + '_WILD_'+ str(j)+' - savgol loss MAE :: ', mean_absolute_error(wild_out, y_wild_filtered))

        
            # kf = KalmanFilter(initial_state_mean=numpy.mean(train_out), n_dim_obs=1)
            # kf = kf.em(y_pred)
            # y_filtered_kf = kf.smooth(y_pred)[0]
            # kf_filter_acc = r2_score(test_out, y_filtered_kf)
            # print(MODEL + '_'+ str(j) + ' - Kalman r² :: ', kf_filter_acc)
            # print(MODEL + '_' + str(j)+' - Kalman loss MAE :: ', mean_absolute_error(test_out, y_filtered_kf))

        acc = r2_score(test_out, y_pred)
        print(MODEL + '_'+ str(j)+' - r² :: ', acc)
        print(MODEL + '_'+ str(j)+' - loss MAE :: ', mean_absolute_error(test_out, y_pred))
        
        acc_wild = r2_score(wild_out, y_wild_pred)
        print(MODEL + '_WILD_'+ str(j)+' - r² :: ', acc_wild)
        print(MODEL + '_WILD_'+ str(j)+' - loss MAE :: ', mean_absolute_error(wild_out, y_wild_pred))

        if verbose!=0:
            print_latitudes(regr_model, poundnet_in, poundnet_date, poundnet_out, 'poundnet', verbose,j)
            print_latitudes(regr_model, large_in, large_date, large_out, 'large', verbose,j)
            print_latitudes(regr_model, medium_in, medium_date, medium_out,'medium', verbose,j)
            print_latitudes(regr_model, small_in, small_date, small_out, 'small', verbose,j)
        if MODEL == 'svm_savgol':
            return filter_acc
        else :
            return acc

def run_condrnn(vars, j, verbose = 0):
    train_data, test_data, poundnet_turtle, large_captive, medium_captive, small_captive, wild_data = prepare_data(vars)
    
    length = 5
    train_c = train_data[length:,:-3]
    train_out = train_data[:, -3]
    train_in_pd = pandas.DataFrame()
    for lag in range(length):
        train_in_pd.loc[:, f'x-{lag + 1}'] = numpy.roll(train_out, lag + 1)
    train_in = train_in_pd.filter(like='x-', axis=1).values[:, :, numpy.newaxis]
    train_in = train_in[length:]
    train_out = train_out[length:]

    poundnet_c = poundnet_turtle[length:,:-3]
    poundnet_out = poundnet_turtle[:, -3]
    poundnet_date = poundnet_turtle[length:, -2]
    poundnet_in_pd = pandas.DataFrame()
    for lag in range(length):
        poundnet_in_pd.loc[:, f'x-{lag + 1}'] = numpy.roll(poundnet_out, lag + 1)
    poundnet_in = poundnet_in_pd.filter(like='x-', axis=1).values[:, :, numpy.newaxis]
    poundnet_in = poundnet_in[length:]
    poundnet_out = poundnet_out[length:]
    
    large_c = large_captive[length:,:-3]
    large_out = large_captive[:, -3]
    large_date = large_captive[length:, -2]
    large_in_pd = pandas.DataFrame()
    for lag in range(length):
        large_in_pd.loc[:, f'x-{lag + 1}'] = numpy.roll(large_out, lag + 1)
    large_in = large_in_pd.filter(like='x-', axis=1).values[:, :, numpy.newaxis]
    large_in = large_in[length:]
    large_out = large_out[length:]

    medium_c = medium_captive[length:,:-3]
    medium_out = medium_captive[:, -3]
    medium_date = medium_captive[length:, -2]
    medium_in_pd = pandas.DataFrame()
    for lag in range(length):
        medium_in_pd.loc[:, f'x-{lag + 1}'] = numpy.roll(medium_out, lag + 1)
    medium_in = medium_in_pd.filter(like='x-', axis=1).values[:, :, numpy.newaxis]
    medium_in = medium_in[length:]
    medium_out = medium_out[length:]

    small_c = small_captive[length:,:-3]
    small_out = small_captive[:, -3]
    small_date = small_captive[length:, -2]
    small_in_pd = pandas.DataFrame()
    for lag in range(length):
        small_in_pd.loc[:, f'x-{lag + 1}'] = numpy.roll(small_out, lag + 1)
    small_in = small_in_pd.filter(like='x-', axis=1).values[:, :, numpy.newaxis]
    small_in = small_in[length:]
    small_out = small_out[length:]

    test_c = test_data[length:,:-3]
    test_out = test_data[:, -3]
    test_in_pd = pandas.DataFrame()
    for lag in range(length):
        test_in_pd.loc[:, f'x-{lag + 1}'] = numpy.roll(test_out, lag + 1)
    test_in = test_in_pd.filter(like='x-', axis=1).values[:, :, numpy.newaxis]
    test_in = test_in[length:]
    test_out = test_out[length:]
    
    wild_c = wild_data[length:,:-3]
    wild_out = wild_data[:,-3]
    wild_in_pd = pandas.DataFrame()
    for lag in range(length):
        wild_in_pd.loc[:, f'x-{lag + 1}'] = numpy.roll(wild_out, lag + 1)
    wild_in = wild_in_pd.filter(like='x-', axis=1).values[:, :, numpy.newaxis]
    wild_in = wild_in[length:]
    wild_out = wild_out[length:]

    train_in = numpy.asarray(train_in).astype('float32')
    train_c = numpy.asarray(train_c).astype('float32')
    train_out = numpy.asarray(train_out).astype('float32')
    test_in = numpy.asarray(test_in).astype('float32')
    test_c = numpy.asarray(test_c).astype('float32')
    test_out = numpy.asarray(test_out).astype('float32')
    large_in = numpy.asarray(large_in).astype('float32')
    large_c = numpy.asarray(large_c).astype('float32')
    large_out = numpy.asarray(large_out).astype('float32')
    poundnet_in = numpy.asarray(poundnet_in).astype('float32')
    poundnet_c = numpy.asarray(poundnet_c).astype('float32')
    poundnet_out = numpy.asarray(poundnet_out).astype('float32')
    medium_in = numpy.asarray(medium_in).astype('float32')
    medium_c = numpy.asarray(medium_c).astype('float32')
    medium_out = numpy.asarray(medium_out).astype('float32')
    small_in = numpy.asarray(small_in).astype('float32')
    small_c = numpy.asarray(small_c).astype('float32')
    small_out = numpy.asarray(small_out).astype('float32')
    wild_in = numpy.asarray(wild_in).astype('float32')
    wild_c = numpy.asarray(wild_c).astype('float32')
    wild_out = numpy.asarray(wild_out).astype('float32')

    regr_model, history = condrnn_vs(train_in, train_c, train_out, test_in, test_out, test_c)
    plot_loss(history, j)
    plot_r2(history, j)

    y_pred = regr_model.predict([test_in, test_c])
    plot_predictions(test_out, y_pred, j)
    y_wild_pred = regr_model.predict([wild_in, wild_c])

    acc = r2_score(test_out, y_pred)
    print(MODEL + '_'+ str(j)+' - r² :: ', acc)
    print(MODEL + '_'+ str(j)+' - loss MAE :: ', mean_absolute_error(test_out, y_pred))
    
    acc_wild = r2_score(wild_out, y_wild_pred)
    print(MODEL + '_WILD_'+ str(j)+' - r² :: ', acc_wild)
    print(MODEL + '_WILD_'+ str(j)+' - loss MAE :: ', mean_absolute_error(wild_out, y_wild_pred))

    if verbose!=0:
        print_latitudes(regr_model, [poundnet_in, poundnet_c], poundnet_date, poundnet_out, 'poundnet', verbose,j)
        print_latitudes(regr_model, [large_in, large_c], large_date, large_out, 'large', verbose,j)
        print_latitudes(regr_model, [medium_in, medium_c], medium_date, medium_out,'medium', verbose,j)
        print_latitudes(regr_model, [small_in, small_c], small_date, small_out, 'small', verbose,j)

    return acc