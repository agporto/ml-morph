# Part of the standard library
import os
import sys
import glob
import argparse
# Not part of the standard library
import dlib

#Parsing arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--test", type=str, default='test.xml',
    help="test data (default=test.xml)", metavar='')
ap.add_argument("-d", "--detector", type=str, default='detector.svm',
    help="trained object detector (default=detector.svm)", metavar='')

args = vars(ap.parse_args())


test_path = os.path.join('./', args['test'])
print("Testing - {}".format(
	dlib.test_simple_object_detector(test_path, args['detector'])))
