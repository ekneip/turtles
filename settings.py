"""
CHOOSE HERE MODEL, TARGET VARIABLE AND VERBOSE
==============================================
possible combinations : 
        vs && dnn, lstm or condRNN
        lat && dnn, lstm, svm-savgol, rf
        lon && svm-savgol
        delta_lat or delta_lon && svm
"""
OUTPUT_VAR = 'delta_lon' # choose from ['vs', 'lat', 'delta_lat', 'lon', 'delta_lon']
MODEL = 'svm' # choose from ['dnn', 'lstm', 'condRNN', 'svm-savgol', 'rf', 'svm']

VERBOSE = 0 # 0, 1 or 2 (0 will only print the MAE and r² for the overall test set and the wild test set, 1 will print for the individuals and 2 will print the individual plots as well)
"""
==============================================
"""

def create_input():
    input_vars = [
        # 'ut', 'vt',   # Don't use this
        'vc', 'uc',
        'SCL',
        'zooc', 'mnkc_epi', 'npp_seapodym', 'mnkc_umeso', 'mnkc_mumeso', 'mnkc_lmeso', 'mnkc_mlmeso', 'mnkc_hmlmeso', 'zeu',
        'T', 'NPP',
        'DayLength', 
        # 'intensity', 'inclination', 'declination',
    ]
    interpolated_vars = [ 'T', 'NPP', 'zooc', 'mnkc_epi', 'npp_seapodym' ]
    interpolated_ranges = [ 10, 25, 50 ]
    interpolated_directions = [ 'left', 'right', 'top', 'bottom' ]

    for v in interpolated_vars:
        for d in interpolated_directions:
            for r in interpolated_ranges:
                input_vars.append('%s_%s_%d' % (v, d, r))
    return input_vars

INPUT_VARIABLES = create_input()

OTHER_VARIABLES = ['mnkc_umeso', 'mnkc_mumeso', 'mnkc_lmeso', 'mnkc_mlmeso', 'mnkc_hmlmeso', 'zeu', 'SCL']

def create_interpolation(var, directions:list, longer=True):
    exp = []
    input_var = [var]
    if longer :
        exp.append(input_var)
    interpolated_directions = directions
    interpolated_ranges = [10, 25, 50]
    for r in interpolated_ranges:
        new_exp=[]
        if longer:
            new_exp = [var]
        for d in interpolated_directions:
            new_exp.append('%s_%s_%d' % (var, d, r))
        exp.append(new_exp)
    return exp

if OUTPUT_VAR == 'lat' or OUTPUT_VAR == 'delta_lat':
    FIRST_EXP_VAR = ['T', 'NPP', 'zooc', 'mnkc_epi', 'npp_seapodym']

    FIRST_EXP_INPUT = [['DayLength']]
    temperature = create_interpolation('T', ['top', 'bottom'])
    for i in range(len(temperature)):
        temperature[i].append('DayLength')
    FIRST_EXP_INPUT += temperature
    for var in FIRST_EXP_VAR:
        FIRST_EXP_INPUT += create_interpolation(var,['top', 'bottom'])
    
    OTHER_VARIABLES.append('vc')

elif OUTPUT_VAR == 'lon' or OUTPUT_VAR == 'delta_lon':
    FIRST_EXP_VAR = ['T', 'NPP', 'zooc', 'mnkc_epi', 'npp_seapodym']

    FIRST_EXP_INPUT = [['DayLength']]
    temperature = create_interpolation('T', ['left', 'right'])
    for i in range(len(temperature)):
        temperature[i].append('DayLength')
    FIRST_EXP_INPUT += temperature
    for var in FIRST_EXP_VAR:
        FIRST_EXP_INPUT += create_interpolation(var,['left', 'right'])

    OTHER_VARIABLES.append('uc')
else : 
    FIRST_EXP_INPUT = []
