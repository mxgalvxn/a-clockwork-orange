# Importing all the functions and libraries
from Z_library import *
from establish_param_monse import *
from manipulation_time_monse import *
    
t0 = time.time()
np.random.seed(42)
print("running")
# Set parameters
parameters = establish_param_time()
parameters["num_sujeto"] = 2
print("Sujeto: ", parameters["num_sujeto"])
print("Tiempo: ", parameters["times"][2])

# Get data and variables
data,parameters = get_data(parameters)
# Filter data through iir filter
filtered_data = iir_filter(data,parameters)
# Order data
ordered_data,parameters = get_ordered_data(filtered_data,parameters)
# Clean data
valid_data = get_clean_data(ordered_data,parameters)

ds_data = downsample_EEG(valid_data,parameters)
# #     # Characteristic vectors and target vectors
X,y,y_options = get_vect_EEG(ds_data,parameters)
# #     # make and test svm
make_n_test_svm_options(X,y_options,parameters)

# AVG data 
#avg_options_data_EEG  = get_avg_options_EEG(valid_data,parameters)
# # Plot AVGS
#plot_EEG_options_avg(avg_options_data_EEG,parameters)

for i in range(len(valid_data["data_img"])):
    print("option: " + parameters["options"][i] + " with "  ,end="")
    print(valid_data["data_img"][i].shape[1], end="")
    print(" trials")

t_final = time.time() - t0
print("time", t_final)
