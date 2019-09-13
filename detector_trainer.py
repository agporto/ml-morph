# Part of the standard library
import os
import sys
import glob
import argparse
# Not part of the standard library
import dlib

#Parsing arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default='train.xml',
    help="training data (default = train.xml)", metavar='')
ap.add_argument("-t", "--test", type=str, default=None,
    help="(optional) test data. if not provided, the model is not tested", metavar='')
ap.add_argument("-o", "--out", type=str, default='detector',
    help="output filename (default = detector)", metavar='')
ap.add_argument("-n", "--n-threads", type=int, default=1,
    help="number of threads to be used (default = 1)", metavar='')
ap.add_argument("-s", "--symmetrical", type=bool, default=False,
    help="(True/False) indicating whether objects are bilaterally symmetrical (default = False)", metavar='')
ap.add_argument("-e", "--epsilon", type=float, default=0.01,
    help="insensitivity parameter (default = 0.01)", metavar='')
ap.add_argument("-c", "--c-param", type=float, default=5,
    help="soft margin parameter C (default =5)", metavar='')
ap.add_argument("-u", "--upsample", type=int, default=0,
    help="upsample limit (default = 0)", metavar='')
ap.add_argument("-w", "--window-size", type=int, default=None,
    help="(optional) detection window size", metavar='')
args = vars(ap.parse_args())

#Setting up the training parameters
options = dlib.simple_object_detector_training_options()
options.add_left_right_image_flips = args['symmetrical']
options.C = args['c_param']
options.num_threads = args['n_threads']
options.be_verbose = True
options.epsilon=args['epsilon']
options.upsample_limit=args['upsample']
if args['window_size'] is not None:
	options.detection_window_size=args['window_size']

#Training the model
train_path = os.path.join('./', args['dataset'])
dlib.train_simple_object_detector(train_path, args['out']+".svm", options)
print("Training - {}".format(
    dlib.test_simple_object_detector(train_path, args['out']+".svm")))

#Testing the model (if test data was provided)
if args['test'] is not None:
    test_path = os.path.join('./', args['test'])
    print("Testing - {}".format(
        dlib.test_simple_object_detector(test_path, args['out']+".svm")))
