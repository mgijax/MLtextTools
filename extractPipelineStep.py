#!/usr/bin/env python3
#
# Script to take a pickled Pipeline object, extract a step (object) from it,
#   and save that object in its own separate pickle file.
#
# Author: Jim Kadin
#
import sys
import pickle
import argparse
#-----------------------------------

def parseCmdLine():
    parser = argparse.ArgumentParser( \
                description='extract a named Pipeline step from pkl file')

    parser.add_argument('--step', dest='pipelineStep', default="classifier",
        help="the step from the Pipeline to extract. " + "Default: classifier")

    parser.add_argument('pkl_input')
    parser.add_argument('pkl_output')

    args = parser.parse_args()
    return args
#----------------------
  
args = parseCmdLine()

def main():

    model = pickle.load(open(args.pkl_input, 'rb'))

    if not model.named_steps.has_key(args.pipelineStep):
        sys.stderr.write("%s has no pipeline step '%s'\n" % \
                            (args.pkl_input, args.pipelineStep))
        exit(5)
    else:
        step = model.named_steps[args.pipelineStep]
        pickle.dump(step, open(args.pkl_output, 'wb'))
# ---------------------------

main()
