# functions to establish parameters for time, freq or time/freq

from  Z_library import *

def establish_param_time():

        # Set parameters 
        Subjects = [16] # amount of subjects
        Trials_per_subject = 1 # amount of trials per subject
        time_test = 6 # secs that the test takes
        time_pre_inter = .1 # secs that we are interested in taking before each trial
        time_inter = 6 # secs that we are interested in
        time_fix = .5 # time that we are interested the fixation
        times = [time_test,time_pre_inter,time_inter,time_fix]
        num_sujeto = 3
        #options_id = [101,102,103,104] # ids that the options have in the marks channel
        #options = ["101","102","103","104"] # options in the same order as the options_id

        options_id = [101,103] # ids that the options have in the marks channel
        options = ["101", "103"] # options in the same order as the options_id

        fix_id = [1] # id that the fixation has in the marks channel

        # parameters for iir filters
        iir_filt = True # True/False
        lowcut = 4 # Hz
        highcut = 40 # Hz
        order = 4

        param_iir_filt = {"iir_filt":iir_filt,"lowcut":lowcut,"highcut":highcut,"order":order}

        # parameters for data cleaning
        clean_filt = True # True/False
        peak_diff_limit = 200
        std_limit = 70

        param_data_cleaning = {"clean_filt":clean_filt,"peak_diff_limit":peak_diff_limit,"std_limit":std_limit}

        # parameters for functional synchrony

        n = 4 # number of cycles for morlet wavelet
        time_window = 1 # secs of window for functional synchrony

        param_func_sync = {"time_window":time_window,"n":n}

        parameters = {"Subjects":Subjects,"num_sujeto":num_sujeto,"Trials_per_subject":Trials_per_subject,"times":times,"options_id":options_id,"options":options,
                "fix_id":fix_id,"param_iir_filt":param_iir_filt,"param_data_cleaning":param_data_cleaning,"param_func_sync":param_func_sync}

        return(parameters)

def establish_param_freq():
        # Set parameters 
        Subjects = [2] # amount of subjects
        Trials_per_subject = 6 # amount of trials per subject
        time_test = 1.5 # secs that the test takes
        time_inter = 1.5 # secs that we are interested in
        time_pre = .5 # time that we are interested in pre-stimuly
        times = [time_test,time_inter,time_pre]
        options_id = [3,4,5,6] # ids that the options have in the marks channel
        options = ["Si","Negativo","Baño","Bebida"] # options in the same order as the options_id
        fix_id = [1] # id that the fixation has in the marks channel

        # parameter for iir filters
        iir_filt = False # True/False
        lowcut = 2 # Hz
        highcut = 60 # Hz
        order = 3

        param_iir_filt = {"iir_filt":iir_filt,"lowcut":lowcut,"highcut":highcut,"order":order}

        # parameter for data cleaning
        clean_filt = True # True/False
        peak_diff_limit = 200
        std_limit =70

        param_data_cleaning = {"clean_filt":clean_filt,"peak_diff_limit":peak_diff_limit,"std_limit":std_limit}

        parameters = {"Subjects":Subjects,"Trials_per_subject":Trials_per_subject,"times":times,"options_id":options_id,"options":options,
                "fix_id":fix_id,"param_iir_filt":param_iir_filt,"param_data_cleaning":param_data_cleaning}

        return(parameters)

def establish_param_tf():
        # Set parameters 
        Subjects = [1] # amount of subjects
        Trials_per_subject = 2 # amount of trials per subject (max 6 for my data)
        time_test = 1.5 # secs that the test takes 
        time_pre_inter = .2 # secs that we are interested in taking before each trial
        time_inter = 1 # secs that we are interested in
        time_fix = .8 # time that we are interested in pre-stimuli
        times = [time_test,time_pre_inter,time_inter,time_fix]
        options_id = [3,4,5,6] # ids that the options have in the marks channel
        options = ["Si","Negativo","Baño","Bebida"] # options in the same order as the options_id
        fix_id = [1] # id that the fixation has in the marks channel

        # parameters for iir filters
        iir_filt = True # True/False
        lowcut = 4 # Hz
        highcut = 40 # Hz
        order = 3

        param_iir_filt = {"iir_filt":iir_filt,"lowcut":lowcut,"highcut":highcut,"order":order}

        # parameters for data cleaning
        clean_filt = True # True/False
        peak_diff_limit = 200
        std_limit = 70

        param_data_cleaning = {"clean_filt":clean_filt,"peak_diff_limit":peak_diff_limit,"std_limit":std_limit}

        t = .15 # number of cycles for morlet wavelet
        t = .3/4
        frequencies = np.arange(2,9,2) # frequencies to measure in ft
        # frequencies = np.arange(2,25,2) # frequencies to measure in ft

        num_freq = len(frequencies)

        parameters = {"Subjects":Subjects,"Trials_per_subject":Trials_per_subject,"times":times,"options_id":options_id,"options":options,
                "fix_id":fix_id,"param_iir_filt":param_iir_filt,"param_data_cleaning":param_data_cleaning,
                "num_freq":num_freq,"morlet_t":t,"frequencies":frequencies}

        return(parameters)
