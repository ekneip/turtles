import pandas
import os
import numpy
from settings import OUTPUT_VAR
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import random

HOME = os.path.expanduser('~')
PATH = ('TRACK_DATA_CNP_LOGGERHEAD/TRACKS')
if not os.path.exists('tmp'):
        os.makedirs('tmp')
CACHE = 'tmp/turtle-dataset-cache.csv'
TEST_CACHE = 'tmp/test_names.csv'

if os.path.isfile(CACHE):
    print('Reusing cache from previous execution... %s' % CACHE)
    dataset = pandas.read_csv(CACHE)

else:
    dataset = None

    for filename in os.listdir(PATH):
        if os.path.splitext(filename) [1] == '.csv':
            print('Reading dataset %s' % filename)
            turtle = pandas.read_csv(os.path.join(PATH, filename),
                                     skipinitialspace=True)
            turtle[['lat_1']] = turtle[['lat']]
            turtle['lat_1'] = turtle.lat_1.shift(-1)
            turtle[['lon_1']] = turtle[['lon']]
            turtle['lon_1'] = turtle.lon_1.shift(-1)

            for lag in range(10):
                turtle_lat_shift = turtle[['lat']]
                turtle_date_shift = turtle[['date']]
                turtle_lon_shift = turtle[['lon']]
                turtle[str(lag+1)+'-lat'] = turtle_lat_shift.shift(-(lag+1))
                turtle[str(lag+1)+"-date"] = turtle_date_shift.shift(-(lag+1))
                turtle[str(lag+1)+"-lon"] = turtle_lon_shift.shift(-(lag+1))

            turtle.drop(turtle.tail(1).index,inplace=True)
            turtle['filename'] = filename

            if dataset is None:
                dataset = turtle
            else:
                dataset = pandas.concat([dataset, turtle], ignore_index=True)



    print('Writing cache... %s' % CACHE)
    dataset.to_csv(CACHE)

DATASET = dataset

def prepare_data(input_vars, lag=None):
    dataset = DATASET
    if OUTPUT_VAR == 'delta_lat' or OUTPUT_VAR == 'delta_lon':
        dataset['delta_lat'] = dataset[str(lag)+'-lat'] - dataset['lat']
        dataset['delta_lon'] = dataset[str(lag)+'-lon'] - dataset['lon']
        vars = input_vars+[OUTPUT_VAR, str(lag)+'-date', 'filename']
    else:
        vars = input_vars+[OUTPUT_VAR, 'date', 'filename']
    dataset = dataset[vars]
    dataset = dataset.dropna()

    if os.path.isfile(TEST_CACHE):
        test_names = pandas.read_csv(TEST_CACHE)
        test_names = test_names["0"].tolist()
        wild_names = ['19598_98.csv', '19604_98.csv', '19602_99.csv', '22153_00.csv', '22328_02.csv', '29060_03.csv']
    else :
        filenames = dataset['filename'].to_numpy()
        filenames = numpy.unique(filenames)
        print_turtles = ['29060_03.csv','40649_04.csv','57152_05.csv','65428_06.csv']
        for turtle in print_turtles:
            filenames = numpy.delete(filenames, numpy.where(filenames == turtle))
        test_names = random.choices(filenames, k=int(0.2*len(filenames)))
        name_df = pandas.DataFrame(test_names)
        name_df.to_csv(TEST_CACHE)

    ct = ColumnTransformer([('stder', StandardScaler(), input_vars)], remainder='passthrough')
    dataset = ct.fit_transform(dataset)
    poundnet_turtle = dataset[numpy.where(dataset[:,-1] == '29060_03.csv')]
    large_captive = dataset[numpy.where(dataset[:,-1] == '40649_04.csv')]
    medium_captive = dataset[numpy.where(dataset[:,-1] == '57152_05.csv')]
    small_captive = dataset[numpy.where(dataset[:,-1] == '65428_06.csv')]

    test_data = dataset[numpy.where((dataset[:,-1] == '29060_03.csv')
                        |(dataset[:,-1] == '40649_04.csv')
                        |(dataset[:,-1] == '57152_05.csv')
                        |(dataset[:,-1] == '65428_06.csv'))]

    train_data = dataset[numpy.where((dataset[:,-1] != '29060_03.csv')*(dataset[:,-1] != '24193_11.csv')*(dataset[:,-1] != '57152_05.csv')*(dataset[:,-1] != '65428_06.csv'))]
    
    test_tuple = (test_data,)
    wild_tuple = ()
    for name in test_names:
        test_tuple += (dataset[numpy.where(dataset[:,-1] == name)],)
        train_data = train_data[numpy.where(train_data[:,-1] != name)]
        if name in wild_names:
            wild_tuple += (dataset[numpy.where(dataset[:,-1] == name)],)
    test_data = numpy.concatenate(test_tuple, axis =0)
    wild_data = numpy.concatenate(wild_tuple, axis=0)

    return (train_data, test_data, poundnet_turtle, large_captive, medium_captive, small_captive, wild_data)