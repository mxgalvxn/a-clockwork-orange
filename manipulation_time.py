# functions to manipulate the data in the time domain

from  Z_library import *

def get_data(parameters):

    Subjects = parameters["Subjects"]
    Sujeto = parameters["num_sujeto"]
    Trials_per_subject = parameters["Trials_per_subject"]
    times = parameters["times"]
    options_id = parameters["options_id"]

    # Get data
    data = []
    for i in range(len(Subjects)):
        for j in range(Trials_per_subject):
            data_T = ebr_file.load_ebr_file("2clasesForms/K" + str(Sujeto) +"N.ebr")
            data.append(np.squeeze(data_T['data']))
        if i == 0:
            sample_freq = data_T['sampling_rate']
            marks_channel = data_T['number_of_channels'] - 1
            channels = data_T['channels'][0:marks_channel-1]

    # Variables saved automatic
    num_channels = len(channels)
    num_options = len(options_id)
    num_experiments = (len(Subjects)*Trials_per_subject)
    samples_per_test = int(times[0]*sample_freq)
    samples_pre_inter = int(times[1]*sample_freq)
    samples_inter = int(times[2]*sample_freq)
    samples_fix = int(times[3]*sample_freq)
    samples_all = samples_inter + samples_pre_inter

    num_info = {"num_experiments":num_experiments,"num_options":num_options,"num_channels":num_channels}

    sample_data = {"sample_freq":sample_freq,"samples_per_test":samples_per_test,"samples_pre_inter":samples_pre_inter,
                "samples_inter":samples_inter,"samples_fix":samples_fix,"samples_all":samples_all}

    parameters["channels"] = channels
    parameters["marks_channel"] = marks_channel
    parameters["num_info"] = num_info
    parameters["sample_data"] = sample_data

    data_t = data[0]
    for i in range(num_experiments-1):
        data_t = np.concatenate((data_t, data[i+1]), axis=1)

    # # Plot marks
    # f, (ax1) = plt.subplots(1, 1,figsize=(15,2))
    # ax1.plot(data[0][marks_channel])
    # ax1.grid(1)
    # ax1.set_xlim(0, len(data[0][marks_channel]))
    # plt.show()

    return(data_t,parameters)

def iir_filter(data,parameters):
    
    marks_channel = parameters["marks_channel"]
    num_info = parameters["num_info"]
    num_channels = num_info["num_channels"]

    param_iir_filt = parameters["param_iir_filt"]
    iir_filt = param_iir_filt["iir_filt"]
    lowcut = param_iir_filt["lowcut"]
    highcut = param_iir_filt["highcut"]
    order = param_iir_filt["order"]
    
    # passing data through iir filter
    if iir_filt == True:

        sample_data = parameters["sample_data"]

        sample_freq = sample_data["sample_freq"]

        b, a = signal.butter(order, [lowcut, highcut], btype='band', fs=sample_freq)

        filtered_data = np.zeros((num_channels+1,len(data[0])))
        for j in range(num_channels):
            filtered_data[j,:] = signal.filtfilt(b, a, data[j])
        filtered_data[-1,:] = data[marks_channel,:]
    else:
        filtered_data = np.zeros((num_channels+1,len(data[0])))
        for j in range(num_channels):
            filtered_data[j,:] = data[j]
        filtered_data[-1,:] = data[marks_channel,:]

    return(filtered_data)

def get_ind(data,parameters):

    marks_channel = parameters["marks_channel"]    
    options_id = parameters["options_id"]    
    fix_id = parameters["fix_id"]
    num_info = parameters["num_info"]

    num_options = num_info["num_options"]

    # save the indices of the marks for each trial/subject
    marks_ind = []
    num_test = []

    for j in range(num_options):
        marks_ind.append(np.where(data[marks_channel-1,:] == options_id[j])[0])
        num_test.append(len(marks_ind[j]))
    fix_ind = np.where(data[marks_channel-1,:] == fix_id)[0]

    # Assumes the amount of tests is the same for all the experiments
    num_fix = len(fix_ind)

    num_info["num_test"] = num_test
    num_info["num_fix"] = num_fix

    parameters["num_info"] = num_info

    inds = {"marks_ind":marks_ind,"fix_ind":fix_ind}

    return(inds,parameters)

def get_tensors_EEG(data,inds,parameters):

    marks_ind = inds["marks_ind"]
    fix_ind = inds["fix_ind"]

    sample_data = parameters["sample_data"]
    num_info = parameters["num_info"]
    num_options = num_info["num_options"]
    num_channels = num_info["num_channels"]
    num_test = num_info["num_test"]
    num_fix = num_info["num_fix"]

    # Tensor for each experiment, channel, option, test, and samples
    # One tensor for silent reading and another one for inner speech
    # Each test "column" is its own "epoch"

    # num_experiments,num_options,num_channels,num_test,num_fix = unpack_num(num_info)
    samples_per_test = sample_data["samples_per_test"]
    samples_pre_inter = sample_data["samples_pre_inter"]
    samples_inter = sample_data["samples_inter"]
    samples_fix = sample_data["samples_fix"]
    samples_all = sample_data["samples_all"]

    # data_img = np.empty([num_options,num_channels,num_test,samples_all],dtype=object)
    data_img = np.empty([num_options],dtype=object)
    data_fix = np.zeros([num_channels,num_fix,samples_fix])

    for j in range(num_options):
        temp = np.zeros([num_channels,num_test[j],samples_all])
        for k in range(num_test[j]):
            temp[:,k,:] = data[0:num_channels, marks_ind[j][k]-samples_pre_inter:marks_ind[j][k]+samples_inter]
        data_img[j] = temp

    for k in range(num_fix):
        data_fix[:,k,:] = data[0:num_channels, fix_ind[k]+samples_per_test*2-samples_fix:fix_ind[k]+samples_per_test*2]
    
    ordered_data = {"data_img":data_img,"data_fix":data_fix}

    return(ordered_data)

def get_diff_peaks_and_std(ordered_data,parameters):

    data_img = ordered_data["data_img"]
    data_fix = ordered_data["data_fix"]
    num_info = parameters["num_info"]

    num_options = num_info["num_options"]

    # get the difference bteween the max and min, the std. For EACH epoch.
    diff_peaks_img = np.empty((num_options),dtype=object)
    std_img = np.empty((num_options),dtype=object)

    for i in range(num_options):
        diff_peaks_img[i] = np.max(data_img[i],axis=2) - np.min(data_img[i],axis=2)
        std_img[i] = np.std(data_img[i],axis=2)
    
    diff_peaks_fix = np.max(data_fix,axis=2) - np.min(data_fix,axis=2)
    std_fix = np.std(data_fix,axis=2)

    diff_std_img = [diff_peaks_img,std_img]
    diff_std_fix = [diff_peaks_fix,std_fix]

    diff_peaks_fix = {"diff_std_img":diff_std_img,"diff_std_fix":diff_std_fix}

    return(diff_peaks_fix)

def get_valid_epochs(peaks_diffs_stds,parameters):

    diff_std_img = peaks_diffs_stds["diff_std_img"]
    diff_std_fix = peaks_diffs_stds["diff_std_fix"]

    param_data_cleaning = parameters["param_data_cleaning"]
    peak_diff_limit = param_data_cleaning["peak_diff_limit"]
    std_limit = param_data_cleaning["std_limit"]
    num_info = parameters["num_info"]
    num_options = num_info["num_options"]

    num_channels = num_info["num_channels"]

    # weed out the noise 
    # Remove epochs where either the diff between peaks and std are above a certain threshold

    # This will give us 1 for each channel when the value is under the threshold, 
    # meaning that for an epoch to be valid, the result from joining the two matrices
    # and summing it up should be equal to twice the amount of channels

    diff_std_sum_img = np.empty((num_options),dtype=object)

    for i in range(num_options):
        diffs_und_lim_img = (diff_std_img[0][i] < peak_diff_limit).astype(int)
        stds_und_lim_img = (diff_std_img[1][i] < std_limit).astype(int)
        diff_std_sum_img[i] = np.sum(diffs_und_lim_img,axis=0) + np.sum(stds_und_lim_img,axis=0)
        diff_std_sum_img[i] = np.floor(diff_std_sum_img[i]/(num_channels*2)).astype(int)

    diffs_und_lim_fix = (diff_std_fix[0] < peak_diff_limit).astype(int)
    std_und_lim_fix = (diff_std_fix[1] < std_limit).astype(int)

    diff_std_sum_fix = np.sum(diffs_und_lim_fix,axis=0) + np.sum(std_und_lim_fix,axis=0)
    diff_std_sum_fix = np.floor(diff_std_sum_fix/(num_channels*2)).astype(int)

    valid_inds = {"valid_inds_img":diff_std_sum_img,"valid_inds_fix":diff_std_sum_fix}

    return(valid_inds)

def get_filtered_EEG(ordered_data,valid_inds,parameters):

    data_img = ordered_data["data_img"]
    data_fix = ordered_data["data_fix"]

    num_info = parameters["num_info"]

    param_data_cleaning = parameters["param_data_cleaning"]
    clean_filt = param_data_cleaning["clean_filt"]

    valid_inds_img = valid_inds["valid_inds_img"]
    valid_inds_fix = valid_inds["valid_inds_fix"]

    if clean_filt == True:

        num_options = num_info["num_options"]

        valid_data_img = np.empty((num_options), dtype=object)
        for i in range(num_options):
            valid_data_img[i] = data_img[i][:,valid_inds_img[i]==1,:]
        valid_data_fix = data_fix[:,valid_inds_fix==1,:]

    else:
        valid_data_img = data_img
        valid_data_fix = data_fix

    valid_data = {"data_img":valid_data_img,"data_fix":valid_data_fix}

    return(valid_data)

# for machine learning

def downsample_EEG(valid_data,parameters):

    num_info = parameters["num_info"]

    num_options = num_info["num_options"]
    num_channels = num_info["num_channels"]

    valid_data_read = copy.deepcopy(valid_data["data_read"])
    valid_data_speech = copy.deepcopy(valid_data["data_speech"])

    for j in range(num_options):
        for k in range(num_channels):
            for ii in range(len(valid_data_read[j,k])):
                valid_data_read[j,k][ii] = valid_data_read[j,k][ii][::4]
            for ii in range(len(valid_data_speech[j,k])):
                valid_data_speech[j,k][ii] = valid_data_speech[j,k][ii][::4]

    ds_data = {"valid_data_read":valid_data_read,"valid_data_speech":valid_data_speech}

    return(ds_data)

def get_vect_EEG(ds_data,parameters):

    ds_read = ds_data["valid_data_read"]
    ds_speech = ds_data["valid_data_speech"]

    num_info = parameters["num_info"]

    num_options = num_info["num_options"]
    num_channels = num_info["num_channels"]

    vect_read = []
    vect_speech = []

    for j in range(num_options):
        for ii in range(len(ds_read[j,0])):
            vect_temp = []
            for k in range(num_channels):
                vect_temp.extend(ds_read[j,k][ii])
            vect_read.append(vect_temp)
        for ii in range(len(ds_speech[j,0])):
            vect_temp = []
            for k in range(num_channels):
                vect_temp.extend(ds_speech[j,k][ii])
            vect_speech.append(vect_temp)
    Y_read = np.zeros(len(vect_read),dtype=int)
    Y_speech = np.ones(len(vect_speech),dtype=int)

    X = {"vect_read":vect_read,"vect_speech":vect_speech}
    y = {"Y_read":Y_read,"Y_speech":Y_speech}

    return(X,y)

def get_vect_EEG_options(ds_data,parameters):

    ds_read = ds_data["valid_data_read"]
    ds_speech = ds_data["valid_data_speech"]

    num_info = parameters["num_info"]

    num_options = num_info["num_options"]
    num_channels = num_info["num_channels"]

    vect_read = []
    vect_speech = []
    Y_read = []
    Y_speech = []


    for j in range(num_options):
        for ii in range(len(ds_read[j,0])):
            vect_temp = []
            for k in range(num_channels):
                vect_temp.extend(ds_read[j,k][ii])
            Y_read.append(j)
            vect_read.append(vect_temp)
        for ii in range(len(ds_speech[j,0])):
            vect_temp = []
            for k in range(num_channels):
                vect_temp.extend(ds_speech[j,k][ii])
            Y_speech.append(j)
            vect_speech.append(vect_temp)

    X = {"vect_read":vect_read,"vect_speech":vect_speech}

    y = {"Y_read":Y_read,"Y_speech":Y_speech}

    return(X,y)

# calculate the avg

def get_avg_options_EEG(valid_data,parameters):

    data_img = valid_data["data_img"]
    data_fix = valid_data["data_fix"]

    num_info = parameters["num_info"]
    num_options = num_info["num_options"]

    # avg through all experiments, and tests, for EACH channel & EACH option

    avg_img = np.empty([num_options],dtype=object)
    for i in range(num_options):
        avg_img[i] = np.average(data_img[i],axis=1)

    avg_fix = np.average(data_fix,axis=1)


    data_avg_option = {"data_option_img":avg_img,"data_fix":avg_fix}

    return(data_avg_option)

# functions to calculate P values from KDE from EEG data

def stat_KDE(avg_data_EEG,parameters):

    data_read = avg_data_EEG["data_read"]
    data_speech = avg_data_EEG["data_speech"]
    data_fix = avg_data_EEG["data_fix"]

    num_info = parameters["num_info"]
    num_channels = num_info["num_channels"]

    channels = parameters["channels"]

    sample_data = parameters["sample_data"]
    sample_freq = sample_data["sample_freq"]
    samples_pre_inter = sample_data["samples_pre_inter"]
    samples_inter = sample_data["samples_inter"]
    samples_all = sample_data["samples_all"]

    data_read_p = np.zeros((num_channels,samples_all))
    data_speech_p = np.zeros(data_read_p.shape)

    data_read_bin = np.zeros(data_read_p.shape)
    data_speech_bin = np.zeros(data_read_p.shape)

    # Calculate the number of rows needed for the subplot grid
    num_rows = (num_channels + 1) // 2  # Ensure it's at least 1 row
    # Set tick locations and labels
    tick_locations = np.arange(0, samples_all + 1, sample_freq/10)  # Every 100 ms (0.1 seconds)
    tick_labels = [f'{(i+1) * -100}' for i in range(int(samples_pre_inter*10/sample_freq)+1)]
    tick_labels.reverse()
    tick_labels_t = [f'{i * 100}' for i in range(int(samples_inter*10/sample_freq))]
    tick_labels.extend(tick_labels_t)
    tick_labels.append(f'{int(samples_inter*10/sample_freq) * 100} ms')

    x_lim = [0,samples_all]
    fig, axs = plt.subplots(num_rows, 2,figsize=(15, 15), sharex=True)
    fig.suptitle('P values for speech & read in respect to pre-stimuli')
    axs = axs.flatten()

    for i in range(num_channels):
        kde_model = gaussian_kde(data_fix[i,:])

        for j in range(samples_all):

            left = 1 - kde_model.integrate_box_1d(-np.inf, data_read[i,j])
            right = 1 - kde_model.integrate_box_1d(data_read[i,j],np.inf)
            if left < right:
                p_value = .50 - left
                if left < .025:
                    data_read_bin[i,j] = 1
            else:
                p_value = -.50 + right
                if right < .025:
                    data_read_bin[i,j] = -1
            data_read_p[i,j] = p_value
            left = 1 - kde_model.integrate_box_1d(-np.inf, data_speech[i,j])
            right = 1 - kde_model.integrate_box_1d(data_speech[i,j],np.inf)
            if left < right:
                p_value = .50 - left
                if left < .025:
                    data_speech_bin[i,j] = 1
            else:
                p_value = -.50 + right
                if right < .025:
                    data_speech_bin[i,j] = -1
            data_speech_p[i,j] = p_value

        axs[i].set_title("Area " + channels[i])
        axs[i].grid(1)
        axs[i].set_xlim(x_lim[0],x_lim[1])
        axs[i].set_xticks(tick_locations, tick_labels,fontsize=8)
        axs[i].plot(data_read_p[i,:],label="read")
        axs[i].plot(data_speech_p[i,:],label="speech")
        axs[i].axhline(y=0.50-.025, color='red', linestyle='--',linewidth=0.5, label = "alpha/2")
        axs[i].axhline(y=-0.50+.025, color='red', linestyle='--',linewidth=0.5)

    axs[i].legend(loc='upper right',fontsize=6)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)

    p_values_data = {"data_read_p":data_read_bin,"data_speech_p":data_speech_bin}

    return(p_values_data)

def stat_word_KDE(avg_word_data_EEG,avg_data_EEG,parameters):
    
    data_read = avg_word_data_EEG["data_word_read"]
    data_speech = avg_word_data_EEG["data_word_speech"]
    data_fix = avg_data_EEG["data_fix"]
    num_info = parameters["num_info"]
    channels = parameters["channels"]
    options = parameters["options"]

    sample_data = parameters["sample_data"]
    sample_freq = sample_data["sample_freq"]
    samples_pre_inter = sample_data["samples_pre_inter"]
    samples_inter = sample_data["samples_inter"]
    samples_all = sample_data["samples_all"]

    num_options = num_info["num_options"]
    num_channels = num_info["num_channels"]

    data_read_p = np.zeros((num_options,num_channels,samples_all))
    data_speech_p = np.zeros(data_read_p.shape)

    data_read_bin = np.zeros(data_read_p.shape)
    data_speech_bin = np.zeros(data_read_p.shape)

    # Calculate the number of rows needed for the subplot grid
    num_rows = (num_channels + 1) // 2  # Ensure it's at least 1 row
    # Set tick locations and labels
    tick_locations = np.arange(0, samples_all + 1, sample_freq/10)  # Every 100 ms (0.1 seconds)
    tick_labels = [f'{(i+1) * -100}' for i in range(int(samples_pre_inter*10/sample_freq)+1)]
    tick_labels.reverse()
    tick_labels_t = [f'{i * 100}' for i in range(int(samples_inter*10/sample_freq))]
    tick_labels.extend(tick_labels_t)
    tick_labels.append(f'{int(samples_inter*10/sample_freq) * 100} ms')

    x_lim = [0,samples_all]

    fig, axs = plt.subplots(num_rows, 2,figsize=(15, 15), sharex=True)
    fig.suptitle('P values for reading the options in respect to pre-stimuli')
    axs = axs.flatten()

    for i in range(num_channels):
        kde_model = gaussian_kde(data_fix[i,:])

        for j in range(num_options):
            
            for k in range(samples_all):
                left = 1 - kde_model.integrate_box_1d(-np.inf, data_read[j,i,k])
                right = 1 - kde_model.integrate_box_1d(data_read[j,i,k],np.inf)
                p_value = min(left,right)
                if left < right:
                    p_value = .50 - left
                    if left < .025:
                        data_read_bin[j,i,k] = 1
                else:
                    p_value = -.50 + right
                    if right < .025:
                        data_read_bin[j,i,k] = -1
                data_read_p[j,i,k] = p_value

                left = 1 - kde_model.integrate_box_1d(-np.inf, data_speech[j,i,k])
                right = 1 - kde_model.integrate_box_1d(data_speech[j,i,k],np.inf)
                p_value = min(left,right)
                if left < right:
                    p_value = .50 - left
                    if left < .025:
                        data_speech_bin[j,i,k] = 1
                else:
                    p_value = -.50 + right
                    if right < .025:
                        data_speech_bin[j,i,k] = -1
                data_speech_p[j,i,k] = p_value

            axs[i].plot(data_read_p[j,i,:],label=options[j])
        axs[i].set_title("Area " + channels[i])
        axs[i].grid(1)
        axs[i].set_xlim(x_lim[0],x_lim[1])
        axs[i].set_xticks(tick_locations, tick_labels,fontsize=8)
        axs[i].axhline(y=0.50-.025, color='red', linestyle='--',linewidth=0.5, label = "alpha/2")
        axs[i].axhline(y=-0.50+.025, color='red', linestyle='--',linewidth=0.5)

    axs[i].legend(loc='upper right',fontsize=6)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)

    fig, axs = plt.subplots(num_rows, 2,figsize=(15, 15), sharex=True)
    fig.suptitle('P values for inner speaking the options in respect to pre-stimuli')
    axs = axs.flatten()

    for i in range(num_channels):

        for j in range(num_options):
            axs[i].plot(data_speech_p[j,i,:],label=options[j])
        axs[i].set_title("Area " + channels[i])
        axs[i].grid(1)
        axs[i].set_xlim(x_lim[0],x_lim[1])
        axs[i].set_xticks(tick_locations, tick_labels,fontsize=8)
        axs[i].axhline(y=0.50-.025, color='red', linestyle='--',linewidth=0.5, label = "alpha/2")
        axs[i].axhline(y=-0.50+.025, color='red', linestyle='--',linewidth=0.5)
    axs[i].legend(loc='upper right',fontsize=6)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)

    p_values_data = {"data_read_p":data_read_bin,"data_speech_p":data_speech_bin}

    return(p_values_data)

def stat_imshow(p_values_data,parameters):

    data_read_p = p_values_data["data_read_p"]
    data_speech_p = p_values_data["data_speech_p"]

    channels = parameters["channels"]
    sample_data = parameters["sample_data"]

    sample_freq = sample_data["sample_freq"]
    samples_pre_inter = sample_data["samples_pre_inter"]
    samples_inter = sample_data["samples_inter"]
    samples_all = sample_data["samples_all"]

    # Set tick locations and labels
    tick_locations = np.arange(0, samples_all + 2, sample_freq/10)  # Every 100 ms (0.1 seconds)
    tick_labels = [f'{(i+1) * -100}' for i in range(int(np.ceil(samples_pre_inter*10/sample_freq)))]
    tick_labels.reverse()
    tick_labels_t = [f'{i * 100}' for i in range(int(np.ceil(samples_inter*10/sample_freq))+1)]
    tick_labels.extend(tick_labels_t)

    fig, axs = plt.subplots(2, 1,figsize=(15, 15), sharex=True)
    axs = axs.flatten()

    # Display the 2D array as an image using imshow
    im1 = axs[0].imshow(data_read_p, cmap='viridis', interpolation='nearest', aspect='auto')
    axs[0].set_title('data read')
    axs[0].set_yticks(np.arange(len(channels)))
    axs[0].set_yticklabels(channels)
    axs[0].grid(1)
    axs[0].set_xticks(tick_locations, tick_labels,fontsize=8)

    divider = make_axes_locatable(axs[0])
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax1)

    im1 = axs[1].imshow(data_speech_p, cmap='viridis', interpolation='nearest', aspect='auto')
    axs[1].set_title('data speech')
    axs[1].set_yticks(np.arange(len(channels)))
    axs[1].set_yticklabels(channels)
    axs[1].grid(1)
    axs[1].set_xticks(tick_locations, tick_labels,fontsize=8)

    divider = make_axes_locatable(axs[1])

    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax1)

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)

def stat_word_imshow(p_values_data,parameters):

    data_read_p = p_values_data["data_read_p"]
    data_speech_p = p_values_data["data_speech_p"]

    num_info = parameters["num_info"]
    channels = parameters["channels"]
    sample_data = parameters["sample_data"]
    options = parameters["options"]

    num_options = num_info["num_options"]
    sample_freq = sample_data["sample_freq"]
    samples_pre_inter = sample_data["samples_pre_inter"]
    samples_inter = sample_data["samples_inter"]
    samples_all = sample_data["samples_all"]

    # Set tick locations and labels
    tick_locations = np.arange(0, samples_all + 2, sample_freq/10)  # Every 100 ms (0.1 seconds)
    tick_labels = [f'{(i+1) * -100}' for i in range(int(np.ceil(samples_pre_inter*10/sample_freq)))]
    tick_labels.reverse()
    tick_labels_t = [f'{i * 100}' for i in range(int(np.ceil(samples_inter*10/sample_freq))+1)]
    tick_labels.extend(tick_labels_t)

    fig, axs = plt.subplots(num_options, 1,figsize=(15, 15), sharex=True)
    axs = axs.flatten()

    for i in range(num_options):

        # Display the 2D array as an image using imshow
        im1 = axs[i].imshow(data_read_p[i,:], cmap='viridis', interpolation='nearest', aspect='auto')
        axs[i].set_title(options[i])
        axs[i].set_yticks(np.arange(len(channels)))
        axs[i].set_yticklabels(channels)
        axs[i].grid(1)
        axs[i].set_xticks(tick_locations, tick_labels,fontsize=8)

        divider = make_axes_locatable(axs[0])
        cax1 = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im1, cax=cax1)

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)

    fig, axs = plt.subplots(num_options, 1,figsize=(15, 15), sharex=True)
    axs = axs.flatten()

    for i in range(num_options):

        # Display the 2D array as an image using imshow
        im1 = axs[i].imshow(data_speech_p[i,:], cmap='viridis', interpolation='nearest', aspect='auto')
        axs[i].set_title(options[i])
        axs[i].set_yticks(np.arange(len(channels)))
        axs[i].set_yticklabels(channels)
        axs[i].grid(1)
        axs[i].set_xticks(tick_locations, tick_labels,fontsize=8)

        divider = make_axes_locatable(axs[0])
        cax1 = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im1, cax=cax1)

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)

def painted_EEG(avg_data_EEG,p_values_data,parameters):

    data_read = avg_data_EEG["data_read"]
    data_speech = avg_data_EEG["data_speech"]
    data_fix = avg_data_EEG["data_fix"]

    data_read_p = p_values_data["data_read_p"]
    data_speech_p = p_values_data["data_speech_p"]

    num_info = parameters["num_info"]
    channels = parameters["channels"]
    sample_data = parameters["sample_data"]

    num_experiments = num_info["num_experiments"]
    num_options = num_info["num_options"]
    num_channels = num_info["num_channels"]
    num_test = num_info["num_test"]
    num_fix = num_info["num_fix"]

    sample_freq = sample_data["sample_freq"]
    samples_pre_inter = sample_data["samples_pre_inter"]
    samples_inter = sample_data["samples_inter"]
    samples_all = sample_data["samples_all"]

    # Calculate the number of rows needed for the subplot grid
    num_rows = (num_channels + 1) // 2  # Ensure it's at least 1 row

    fig, axs = plt.subplots(num_rows, 2, figsize=(15, 15), sharex=True)
    fig.suptitle('AVG EEG data for reading')

    # Set tick locations and labels
    tick_locations = np.arange(0, samples_all + 1, sample_freq/10)  # Every 100 ms (0.1 seconds)
    tick_labels = [f'{(i+1) * -100}' for i in range(int(samples_pre_inter*10/sample_freq)+1)]
    tick_labels.reverse()
    tick_labels_t = [f'{i * 100}' for i in range(int(samples_inter*10/sample_freq))]
    tick_labels.extend(tick_labels_t)
    tick_labels.append(f'{int(samples_inter*10/sample_freq) * 100} ms')

    # Flatten the axs array so we can iterate through it in a linear fashion
    axs = axs.flatten()
    x_lim = [0,samples_all]

    for i in range(num_channels):                                      
        absolute_max = np.max(abs(data_read[i,:]))

        # black if 0, red if 1, green if -1
        for j in range(len(data_read[i,:])):
            if data_read_p[i,j] == 1:
                color = "red"
            elif data_read_p[i,j] == -1:
                color = "green"
            else:
                color = "black"
            # color = 'red' if data_read_p[i,j] == 1 elif color = 'red' if data_read_p[i,j] == 1 else 'black'
            axs[i].plot([j, j + 1], [data_read[i,j], data_read[i,j]], color=color, linewidth=2)

        # axs[i].plot(data_read[i,:], label="Silent Reading")
        axs[i].set_title("Area " + channels[i])
        axs[i].grid(1)
        axs[i].set_xlim(x_lim[0],x_lim[1])
        axs[i].set_ylim(-absolute_max,absolute_max)
        axs[i].set_xticks(tick_locations, tick_labels,fontsize=8)
    # axs[i].legend(loc='upper right',fontsize=6)
    axs[num_channels-1].set_xlabel("time")
    axs[num_channels-2].set_xlabel("time") 

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)

    fig, axs = plt.subplots(num_rows, 2, figsize=(15, 15), sharex=True)
    fig.suptitle('AVG EEG data for speech')

    # Flatten the axs array so we can iterate through it in a linear fashion
    axs = axs.flatten()

    for i in range(num_channels):                                      
        absolute_max = np.max(abs(data_speech[i,:]))

        # black if 0, red if 1, green if -1
        for j in range(len(data_speech[i,:])):
            if data_speech_p[i,j] == 1:
                color = "red"
            elif data_speech_p[i,j] == -1:
                color = "green"
            else:
                color = "black"
            axs[i].plot([j, j + 1], [data_speech[i,j], data_speech[i,j]], color=color, linewidth=2)

        axs[i].set_title("Area " + channels[i])
        axs[i].grid(1)
        axs[i].set_xlim(x_lim[0],x_lim[1])
        axs[i].set_ylim(-absolute_max,absolute_max)
        axs[i].set_xticks(tick_locations, tick_labels,fontsize=8)
    axs[num_channels-1].set_xlabel("time")
    axs[num_channels-2].set_xlabel("time") 

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)

# functions that call other functions

def get_clean_data(ordered_data,parameters):

    peaks_diffs_stds = get_diff_peaks_and_std(ordered_data,parameters)
    valid_inds = get_valid_epochs(peaks_diffs_stds,parameters)
    valid_data = get_filtered_EEG(ordered_data,valid_inds,parameters)

    return(valid_data)

def get_ordered_data(filtered_data,parameters):
    
    inds,parameters = get_ind(filtered_data,parameters)
    ordered_data = get_tensors_EEG(filtered_data,inds,parameters)

    return(ordered_data,parameters)

# functions to plot

def plot_EEG_options_avg(avg_img,parameters):

    data_option_img = avg_img["data_option_img"]
    data_fix = avg_img["data_fix"]

    channels = parameters["channels"]
    options = parameters["options"]

    num_info = parameters["num_info"]
    num_options = num_info["num_options"]
    num_channels = num_info["num_channels"]

    sample_data = parameters["sample_data"]
    sample_freq = sample_data["sample_freq"]
    samples_pre_inter = sample_data["samples_pre_inter"]
    samples_inter = sample_data["samples_inter"]
    samples_all = sample_data["samples_all"]

    Sujeto = parameters["num_sujeto"]
    # Calculate the number of rows needed for the subplot grid
    num_rows = (num_channels + 1) // 2  # Ensure it's at least 1 row

    fig, axs = plt.subplots(num_rows, 2, figsize=(15, 15), sharex=True)
    fig.suptitle('AVG EEG data Sujeto: '+  str(Sujeto))

    # Set tick locations and labels
    tick_locations = np.arange(0, samples_all + 1, sample_freq/10)  # Every 100 ms (0.1 seconds)
    tick_labels = [f'{(i+1) * -100}' for i in range(int(np.floor(samples_pre_inter*10/sample_freq))+1)]
    tick_labels.reverse()
    tick_labels_t = [f'{i * 100}' for i in range(int(np.ceil(samples_inter*10/sample_freq))+1)]
    tick_labels.extend(tick_labels_t)

    # Flatten the axs array so we can iterate through it in a linear fashion
    axs = axs.flatten()
    x_lim = [0,samples_inter]
    for i in range(num_channels):
        max = 0
        for j in range(2):
            axs[i].plot(data_option_img[j][i,:], label = options[j])
            absolute_max = np.max(abs(data_option_img[j][i,:]))
            if absolute_max > max:
                max = absolute_max
        #axs[i].plot(data_fix[i,:],label = "fixation")
        axs[i].set_title("Area " + channels[i])
        axs[i].grid(1)
        axs[i].set_ylim(-max,max)
        axs[i].set_xlim(x_lim[0],x_lim[1])
        axs[i].set_xticks(tick_locations, tick_labels,fontsize=8)
    axs[num_channels-1].set_xlabel("time")
    axs[num_channels-2].set_xlabel("time")
    axs[i].legend(loc='upper right',fontsize=6)

    plt.tight_layout()
    plt.show(block = False)
    plt.pause(0.001)

def make_n_test_svm_options(X,y,parameters):

    vect_read_word = X["vect_read"]

    Y_read_word = y["Y_read"]

    options = parameters["options"]

    kernels = ["linear"]

    c = np.logspace(np.log10(.01),np.log10(100),5)
    g = c

    results_read = np.zeros([len(kernels),len(c),len(g)])

    for i in range(len(kernels)):
        for j in range(len(c)):
            for k in range(len(g)):
                svm_classifier = svm.SVC(kernel=kernels[i],C=c[j],gamma=g[k],class_weight="balanced")
                cv_scores = cross_val_score(svm_classifier, vect_read_word, Y_read_word, cv=4)
                results_read[i,j,k] = np.mean(cv_scores)
                # print(kernels[i],c[j],g[k],results_read[i,j,k])

    inds = np.unravel_index(np.argmax(results_read), results_read.shape)

    print("kernel: ",kernels[inds[0]], ",c: ",c[inds[1]],",gamma: ",g[inds[2]])
    print(np.max(results_read))

    # svm_classifier = svm.SVC(kernel=kernels[inds[0]],C=c[inds[1]],gamma=g[inds[2]])
    # predicted = cross_val_predict(svm_classifier, vect_read_word, Y_read_word, cv=4)
    # conf_matrix = confusion_matrix(Y_read_word, predicted)
    # conf_matrix_percentage = 100 * conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    # # Plot the confusion matrix as a heatmap for better visualization
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(conf_matrix_percentage, annot=True, fmt='.2f', cmap='Blues', xticklabels=options, yticklabels=options)
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.title('Normalized Confusion Matrix read for best svm')
    # plt.show(block=False)
    # plt.pause(0.001)

    # svm_classifier = svm.SVC(kernel=kernels[inds[0]],C=c[inds[1]],gamma=g[inds[2]])
    # predicted = cross_val_predict(svm_classifier, vect_speech_word, Y_speech_word, cv=4)
    # conf_matrix = confusion_matrix(Y_speech_word, predicted)
    # conf_matrix_percentage = 100 * conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    # # Plot the confusion matrix as a heatmap for better visualization
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(conf_matrix_percentage, annot=True, fmt='.2f', cmap='Blues', xticklabels=options, yticklabels=options)
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.title('Normalized Confusion Matrix speech for best svm')
    # plt.show(block=False)
    # plt.pause(0.001)

    predicted = cross_val_predict(svm_classifier, vect_read_word, Y_read_word, cv=4)

    # # Calculate the confusion matrix
    conf_matrix = confusion_matrix(Y_read_word, predicted)

    print(conf_matrix)
    # conf_matrix_percentage = 100 * conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    # # Plot the confusion matrix as a heatmap for better visualization
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(conf_matrix_percentage, annot=True, fmt='.2f', cmap='Blues', xticklabels=options, yticklabels=options)
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.title('Confusion Matrix read')
    # plt.show(block=False)
    # plt.pause(0.001)

    recalls = []
    for i in range(len(conf_matrix)):
        true_positive = conf_matrix[i, i]
        false_negative = sum(conf_matrix[i, :]) - true_positive
        recall = true_positive / (true_positive + false_negative)
        recalls.append(recall)

    print(recalls)

    # Print the recall for each class
    for i in range(len(recalls)):
        print("Recall Class, " + options[i])
        print(str(recalls[i].round(2)))

    # # Create an SVM classifier
    # svm_classifier = svm.SVC(kernel='rbf')  # You can choose different kernels, such as 'linear', 'rbf', etc.

    # # Fit the classifier on the training data
    # cv_scores = cross_val_score(svm_classifier, vect_speech_word, Y_speech_word, cv=4)

    # print("speech")
    # # Print the cross-validated accuracy scores
    # print("Cross-validated Accuracy Scores:", cv_scores)

    # # Calculate the mean accuracy
    # mean_accuracy = np.mean(cv_scores)
    # print("Mean Accuracy:", mean_accuracy)

    # predicted = cross_val_predict(svm_classifier, vect_speech_word, Y_speech_word, cv=4)

    # # Calculate the confusion matrix
    # conf_matrix = confusion_matrix(Y_speech_word, predicted)

    # conf_matrix_percentage = 100 * conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    # # Plot the confusion matrix as a heatmap for better visualization
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(conf_matrix_percentage, annot=True, fmt='.2f', cmap='Blues', xticklabels=options, yticklabels=options)
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.title('Confusion Matrix speech')
    # plt.show(block=False)
    # plt.pause(0.001)

    # recalls = []
    # for i in range(len(conf_matrix)):
    #     true_positive = conf_matrix[i, i]
    #     false_negative = sum(conf_matrix[i, :]) - true_positive
    #     recall = true_positive / (true_positive + false_negative)
    #     recalls.append(recall)

    # for i in range(len(recalls)):
    #     print("Recall Class, " + options[i])
    #     print(str(recalls[i].round(2)))

def downsample_EEG(valid_data,parameters):

    data_read = valid_data["data_img"]
    data_fix = valid_data["data_fix"]

    num_info = parameters["num_info"]
    num_options = num_info["num_options"]

    ds_read = np.empty([num_options],dtype=object)
    ds_speech = np.empty([num_options],dtype=object)

    for i in range(num_options):
        ds_read[i] = data_read[i][:,:,::4]
    ds_fix = data_fix[:,:,::4]

    ds_data = {"data_read":ds_read,"data_fix":ds_fix}

    return(ds_data)

def get_vect_EEG(ds_data,parameters):

    data_read = ds_data["data_read"]

    num_info = parameters["num_info"]

    num_options = num_info["num_options"]
    num_channels = num_info["num_channels"]

    vect_read = []
    vect_speech = []

    Y_read_options = []
    Y_speech_options = []

    for i in range(num_options):

        for j in range(len(data_read[i][0])):
            vect_temp = []
            for k in range(num_channels):
                vect_temp.extend(data_read[i][k,j,:])
            vect_read.append(vect_temp)
            Y_read_options.append(i)

    Y_read = np.zeros(len(vect_read),dtype=int)

    X = {"vect_read":vect_read}
    y = {"Y_read":Y_read}
    y_options = {"Y_read":Y_read_options}

    return(X,y,y_options)


