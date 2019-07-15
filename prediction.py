import argparse
import os
import utils


ap = argparse.ArgumentParser()
ap.add_argument('-i','--input-dir', type=str, default='pred', help="input directory (default = pred)", metavar='')
ap.add_argument('-d','--detector', type=str, default='detector.svm', help="trained object detection model (default = detector.svm)", metavar='')
ap.add_argument('-p','--predictor', type=str, default='predictor.dat', help="trained shape prediction model (default = predictor.dat)", metavar='')
ap.add_argument('-o','--out-file', type=str, default='output.xml', help="output file name (default = output.xml)", metavar='')
ap.add_argument('-u','--upsample-limit', type=int, default=0, help="upsample limit (default= 0 ; max = 2)", metavar='')
ap.add_argument('-t','--threshold', type=float, default=0, help="detector's confidence threshold for outputting an object (default= 0)", metavar='')
ap.add_argument('-l','--ignore-list', nargs='*', type=int, default=None, help=" (optional) prevents landmarks of choice from being output", metavar='')






args = vars(ap.parse_args())

utils.predictions_to_xml(args['detector'],args['predictor'], dir=args['input_dir'],upsample=args['upsample_limit'],threshold=args['threshold'],ignore=args['ignore_list'],out_file=args['out_file'])

