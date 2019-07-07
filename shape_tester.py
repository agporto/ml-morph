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
    help="test data: xml filename", metavar='')
ap.add_argument("-p", "--predictor", type=str, default='predictor.dat',
    help="trained shape predictor", metavar='')

args = vars(ap.parse_args())


test_path = os.path.join('./', args['test'])
print("Testing error (mean pixel deviation): {}".format(
	dlib.test_shape_predictor(test_path, args['predictor'])))