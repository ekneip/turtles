import os
from help_functions import run_condrnn, run_model
from settings import FIRST_EXP_INPUT, INPUT_VARIABLES, MODEL, OTHER_VARIABLES, OUTPUT_VAR, VERBOSE
from matplotlib import pyplot as plt
import sys

orig_stdout = sys.stdout
f = open(MODEL+ '_results_'+ OUTPUT_VAR+'.txt', 'a')
sys.stdout = f 


def all_var(lag = None):
    print('INPUT VARIABLES USED ::')
    print(INPUT_VARIABLES)
    print('================RESULTS================')
    if MODEL == 'condRNN':
        acc= run_condrnn(INPUT_VARIABLES, 0, VERBOSE)
    else:
        acc = run_model(INPUT_VARIABLES, 0, VERBOSE, lag)
    print('=======================================')
    return acc

def exp_1(lag=None):
    accs = {}
    for i in range(len(FIRST_EXP_INPUT)):
        print('INPUT VARIABLES USED ::')
        print(FIRST_EXP_INPUT[i])
        print('================RESULTS================')
        acc = run_model(FIRST_EXP_INPUT[i], i+1, VERBOSE, lag)
        accs[i+1] = acc
        print('=======================================')
    return accs

def exp_2():
    accs = {}
    # mnkc_epi 21; zooplankton 17; temp+daylength 5
    if OUTPUT_VAR == 'lat':
        variables = FIRST_EXP_INPUT[20]
        print('INPUT VARIABLES USED ::')
        print(variables)
        print('================RESULTS================')
        acc = run_model(variables, 21, VERBOSE)
        accs[0] = acc
        print('=======================================')
        variables += FIRST_EXP_INPUT[16]
        print('INPUT VARIABLES USED ::')
        print(variables)
        print('================RESULTS================')
        acc = run_model(variables, 101, VERBOSE)
        accs[1] = acc
        print('=======================================')
        variables += FIRST_EXP_INPUT[4]
        print('INPUT VARIABLES USED ::')
        print(variables)
        print('================RESULTS================')
        acc = run_model(variables, 102, VERBOSE)
        accs[2] = acc
        print('=======================================')
    else :
        variables = FIRST_EXP_INPUT[20]
        print('INPUT VARIABLES USED ::')
        print(variables)
        print('================RESULTS================')
        acc = run_model(variables, 21, VERBOSE)
        accs[0] = acc
        print('=======================================')
        variables += FIRST_EXP_INPUT[16]
        print('INPUT VARIABLES USED ::')
        print(variables)
        print('================RESULTS================')
        acc = run_model(variables, 201, VERBOSE)
        accs[101] = acc
        print('=======================================')
        variables += FIRST_EXP_INPUT[24]
        print('INPUT VARIABLES USED ::')
        print(variables)
        print('================RESULTS================')
        acc = run_model(variables, 202, VERBOSE)
        accs[102] = acc
        print('=======================================')
    if OUTPUT_VAR == 'lat':
        order_var = [3,1,4,0,2,5,6,7]
    else :
        order_var = [1,4,0,3,2,5,7,6]
    for i in range(len(OTHER_VARIABLES)):
        variables.append(OTHER_VARIABLES[order_var[i]])
        print('INPUT VARIABLES USED ::')
        print(variables)
        print('================RESULTS================')
        acc = run_model(variables, 203+i, VERBOSE)
        accs[203+i] = acc
        print('=======================================')

    # for i in range(len(OTHER_VARIABLES)):
    #     acc = run_model(variables + [OTHER_VARIABLES[i]], 102+i, VERBOSE)
    #     accs[102+i] = acc

    # sorted_d = sorted(accs.items(), key=operator.itemgetter(1))
    # print(sorted_d)

    
    lists = sorted(accs.items()) # sorted by key, return a list of tuples
    _, y = zip(*lists) 
    if OUTPUT_VAR == 'lat':
        x = [FIRST_EXP_INPUT[20][0]] +  [FIRST_EXP_INPUT[16][0]] + [FIRST_EXP_INPUT[4][0] + ' & DayLength']+ [OTHER_VARIABLES[order_var[i]] for i in range(len(OTHER_VARIABLES))]
    else : 
        x = [FIRST_EXP_INPUT[20][0]] +  [FIRST_EXP_INPUT[16][0]] + [FIRST_EXP_INPUT[24][0]]+ [OTHER_VARIABLES[order_var[i]] for i in range(len(OTHER_VARIABLES))]

    plt.plot(x,y)
    plt.xlabel('Input variables added to the epipelagic micronekton')
    plt.ylabel('R² score')
    plt.grid()
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    if not os.path.exists('images_thesis/'+MODEL+'/'):
        os.makedirs('images_thesis/'+MODEL+'/')
    plt.savefig('images_thesis/'+MODEL+'/'+OUTPUT_VAR+'_exp2')

def plot_scores_delta():
    accs = {}
    for lag in range(1,11):
        acc = all_var(lag)
        accs[lag] = acc
    listed_accs = sorted(accs.items())
    x, y = zip(*listed_accs)
    plt.xlabel('Days')
    plt.ylabel('R² score')
    plt.grid()
    plt.plot(x, y)
    plt.tight_layout()
    if not os.path.exists('images_thesis/'+MODEL+'/'):
        os.makedirs('images_thesis/'+MODEL+'/')
    plt.savefig('images_thesis/'+MODEL+'/'+OUTPUT_VAR+'_delta_scores')

def exp_1_delta():
    for lag in [1,10]:
        exp_1(lag)

def plot_var_delta():
    if OUTPUT_VAR == 'delta_lat':
        first_variable = FIRST_EXP_INPUT[4]
        second_variable = [OTHER_VARIABLES[len(OTHER_VARIABLES)-1]]
    else :
        first_variable = ['SCL']
        second_variable = [OTHER_VARIABLES[len(OTHER_VARIABLES)-1]]
    accs_allvar = {}
    accs_first = {}
    accs_second = {}
    for lag in range(1,11):
        acc = all_var(lag)
        accs_allvar[lag] = acc
        print('INPUT VARIABLES USED ::')
        print(first_variable)
        print('================RESULTS================')
        acc = run_model(first_variable, 500+lag, VERBOSE, lag)
        print('=======================================')
        accs_first[lag] = acc
        print('INPUT VARIABLES USED ::')
        print(second_variable)
        print('================RESULTS================')
        acc = run_model(second_variable, 600+lag, VERBOSE, lag)
        print('=======================================')
        accs_second[lag] = acc
    listed_accs = sorted(accs_allvar.items())
    x, y = zip(*listed_accs)
    listed_1accs = sorted(accs_first.items())
    x_1, y_1 = zip(*listed_1accs)
    listed_2accs = sorted(accs_second.items())
    x_2, y_2 = zip(*listed_2accs)
    plt.xlabel('Days')
    plt.ylabel('R² score')
    plt.plot(x, y, label='Using all environmental variables')
    if OUTPUT_VAR == 'delta_lat':
        plt.plot(x_1, y_1, label='Using only the North-South T gradient and DayLength', color = 'purple')
        plt.plot(x_2, y_2, label='Using only the vc', color = 'chocolate')
    else :
        plt.plot(x_1, y_1, label='Using only the size of the turtle', color = 'purple')
        plt.plot(x_2, y_2, label='Using only the uc', color = 'chocolate')
    plt.legend(loc= 'upper right')
    plt.grid()
    plt.tight_layout()
    if not os.path.exists('images_thesis/'+MODEL+'/'):
        os.makedirs('images_thesis/'+MODEL+'/')
    plt.savefig('images_thesis/'+MODEL+'/'+OUTPUT_VAR+'_delta_variables')

"""only applied to lat, lon and vs prediction"""
# all_var() # RUNS THE TRAINING MODEL SELECTED TO PREDICT THE VARIABLE SELECTED WITH ALL ENVIRONMENTAL VARIABLES
"""only applied to lat, lon prediction with svm-savgol model"""
# exp_1() # RUNS THE FIRST FEATURE ANALYSIS
# exp_2() # RUNS THE SECOND FEATURE ANALYSIS
""""only applied to delta_lat, delta_lon prediction with svm model"""
# plot_scores_delta() # RUNS THE ALL_VAR METHOD ON ALL TIME RANGES
# plot_var_delta() # RUNS THE VARIABLE ANALYSIS FOR THE DELTA_LAT AN DELTA_LON TARGET VARIABLES
sys.stdout = orig_stdout
f.close()