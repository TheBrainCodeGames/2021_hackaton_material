import bz_LoadBinary
import numpy as np
import pandas as pd
import scipy.io
import sys
import os


"""
Auxiliary methods for loading the data and saving 
the results from BrainCodeGames 2021 hackaton. 

Methods
-------
load_data (path)
	Loads LFP data for a session

load_ripples_tags (path, fs)
	Loads ripples start and end times (in seconds) for a session.

get_ripples_tags_as_signal (data, ripples, fs)
	Generates a pulse signal representing all the ripples tagged or predicted in a session.

get_score (true_ripples, pred_ripples, threshold=0.1)
	Calculates the precision, recall and F1 metrics for some detected events over the ground truth.

write_results (save_path, session_name, group_number, predictions)
	Saves the predicted ripples into a CSV file.
"""


def __load_info (path):
	try:
		mat = scipy.io.loadmat(path+"/info.mat")
	except:
		print("info.mat file does not exist.")
		sys.exit()

	fs = mat["fs"][0][0]
	exp_name = mat["expName"][0]

	ref_channels = {}
	ref_channels["so"] = mat["so"][0]
	ref_channels["pyr"] = mat["pyr"][0]
	ref_channels["rad"] = mat["rad"][0]
	ref_channels["slm"] = mat["slm"][0]


	if len(mat["chDead"]) <= 0:
		dead_channels = []
	else:
		dead_channels = [x-1 for x in (mat["chDead"][0]).astype(int)]

	return fs, exp_name, ref_channels, dead_channels

def __load_raw_data (path, channels):
	# There is .dat file
	is_dat = any([file.endswith(".dat") for file in os.listdir(path)])


	if is_dat:
		name_dat = os.listdir(path)[np.where([file.endswith(".dat") for file in os.listdir(path)])[0][0]]
		data = bz_LoadBinary.bz_LoadBinary(path+"/"+name_dat, len(channels), channels, 2, False)
	else:
		print('No data found')

	return np.array(data)



def __iou (x, x_array=[]):
	"""Implement the intersection over union (IoU) between x and x_array

	Arguments:
	x -- first segment, numpy array with coordinates (x_ini, x_end)
	x_array -- second segment, numpy array with coordinates (x_array_ini, x_array_end)
	"""

	# Assign variable names to coordinates for clarity
	if len(x_array)>0:
		x_array_ini = x_array[:,0]
		x_array_end = x_array[:,1]
	else:
		return np.array([])

	x_ini = x[0] * np.ones_like(x_array_ini)
	x_end = x[1] * np.ones_like(x_array_ini)

	# Calculate the (xi1, xi2) coordinates of the intersection of x and x_array. Calculate its duration.
	xi1 = np.max([x_ini, x_array_ini], axis=0)
	xi2 = np.min([x_end, x_array_end], axis=0)
	inter_duration = xi2 - xi1

	# Calculate the Union duration by using Formula: Union(A,B) = A + B - Inter(A,B)
	x_duration = x_end-x_ini
	x_array_duration = x_array_end-x_array_ini
	union_duration = x_duration + x_array_duration - inter_duration

	# compute the IoU
	iou = inter_duration / union_duration

	return iou


def load_data (path):
	"""
	Loads the LFP signals from a session.

	Parameters
	----------
	path : str
		Path to the folder where the session data is located.

	Returns
	-------
	data : Numpy array (n x 8)
		Numpy array containing the LFP. It has n rows, one for each 
		timestep, and 8 columns, one per channel.
	fs : int
		Data sampling rate (in Hz).
	session_name : str
		String containing the name of the session.


	Example
	-------
	data, fs, session_name = load_data("/myfolder/braincodegames/data/session1/")

	"""

	# Read info.mat
	fs, session_name, ref_channels, dead_channels = __load_info(path)

	# Read .dat
	channels = list(range(8))
	data = __load_raw_data(path, channels)

	return data, fs, session_name



def load_ripples_tags (path, fs):
	"""
	Loads the start and end times of all the ripples tagged in a session.

	Parameters
	----------
	path : str
		Path to the folder where the session data is located.
	fs : int
		Session data sampling rate (in Hz).

	Returns
	-------
	ripples : Numpy array (n x 2)
		Numpy array with the start and end times of the ripples (in seconds). 
		It has n rows, one for each ripple, and 2 columns, for the start and 
		end times respectively.


	Example
	-------
	ripples_tags = load_ripples_tags("/myfolder/braincodegames/data/session1/", 30000)

	"""

	try:
		dataset = pd.read_csv(path+"/ripples.csv", delimiter=' ', header=0, usecols = ["ripIni", "ripMiddle", "ripEnd", "type", "shank"])
	except:
		print(path+"/ripples.csv file does not exist.")
		sys.exit()

	ripples = dataset.values
	ripples = ripples[np.argsort(ripples, axis=0)[:, 0], :]

	return np.array(ripples)[:,[0, 2]] / fs



def get_ripples_tags_as_signal (data, ripples, fs):
	"""
	Generates a pulse signal representing all the ripples tagged or predicted in a session.

	Parameters
	----------
	data: Numpy array (n x 2)
		Numpy array containing the LFP. It has n rows, one for each 
		timestep, and 8 columns, one per channel.
	ripples : Numpy array (n x 2)
		Numpy array the start and end times of the ripples (in seconds). 
		It has n rows, one for each prediction, and 2 columns, for the 
		start and end times respectively.
	fs : int
		Session data sampling rate (in Hz).

	Returns
	-------
	signal : Numpy array (n x 1)
		Numpy 1D vector containing a squared signal of duration n, as the number 
		of rows in data. There will be a 1 for each timestep that contains a ripple,
		or 0 otherwise.


	Example
	-------
	signal_true = get_ripples_as_signal(data, true_ripples, fs)
	signal_predicted = get_ripples_as_signal(data, pred_ripples, fs)

	"""

	signal = np.zeros(np.shape(data)[0])

	for ripple in (ripples * fs):
		for i in range(int(ripple[0]), int(ripple[1])):
			signal[i] = 1

	return signal



def get_score (true_ripples, pred_ripples, threshold=0.1):
	"""
	Calculates the precision, recall and F1 metrics for some detected events
	over the ground truth.

	Parameters
	----------
	true_ripples : Numpy array (n x 2)
		Numpy array the start and end times of the tagged ripples (in seconds). 
		It has n rows, one for each prediction, and 2 columns, for the 
		start and end times respectively.
	pred_ripples : Numpy array (n x 2)
		Numpy array the start and end times of the detected ripples (in seconds). 
		It has n rows, one for each prediction, and 2 columns, for the 
		start and end times respectively.
	threshold : float
		IoU threshold. Default=0.1 (Optional)

	Returns
	-------
	precision : float
		Number of correct detections over total number of detections.
	recall: float
		Number of true ripples detected over total number of true ripples.
	F1: float
		Harmonic mean of precision and recall.


	Example
	-------
	precision, recall, F1 = get_score(ripples_tags, detected_ripples)

	"""

	true_positives = 0
	false_positives = 0
	false_negatives = np.shape(true_ripples)[0]

	for i_event in range(np.shape(pred_ripples)[0]):
		# Calculate IoU of the pred event with all true events
		ious = __iou(pred_ripples[i_event, :], true_ripples)

		# Get the best IoU
		if len(ious) < 1:
			# There are no ground truths so it is a false positive
			false_positives += 1
			continue

		best_iou_index = np.argmax(ious)
		if ious[best_iou_index] >= threshold:
			# If IoU >= threshold, it is a true positive. Remove from true events to find
			false_negatives -= 1
			true_positives += 1

		else:
			# If not, it is a false positive
			false_positives += 1


	precision = true_positives / (true_positives + false_positives) if true_positives != 0 else 0
	recall = true_positives / (true_positives + false_negatives) if true_positives != 0 else 0
	F1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0


	return precision, recall, F1



def write_results (save_path, session_name, group_number, predictions):
	"""
	Writes the predictions for a session into a CSV file.

	Parameters
	----------
	save_path : str
		Path to the folder where the file will be created.
	session_name : str
		Name of the session.
	group_number : int
		Identification number of the participating group.
	predictions : Numpy array (n x 2)
		Numpy array the start and end times of the predicted ripples 
		(in seconds). It has n rows, one for each prediction, and 2 
		columns, for the start and end times respectively.

	Example
	-------
	write_results("/myfolder/braincodegames/results/", "session1", 7, my_ripple_predictions)

	"""

	filename = save_path + "/Group" + str(group_number) + "_" + session_name + ".csv" 
	np.savetxt(filename, predictions, delimiter=" ")

