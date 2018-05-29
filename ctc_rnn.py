"""
Module for running on the command line
Three options are given
-- transform : feature transforming and store in the specified folder
-- validation: running validation for the model defined in the specified directory
-- train: train the input model
"""

import argparse
import os
import sys
import featext
from featext import Phonemefeat
import train

def parse_args():
    """
    Parse the command line arguments
    """
    parser = argparse.ArgumentParser()
    # feature extraction
    parser.add_argument("-fT",
                        "--feature-transform",
                        help=("Transform the audios in TIMIT directory " +
                              "to mfccs"), action="store_true")
    parser.add_argument("-timitD",
                        "--timit-dir",
                        help=("The path of the timit directory. " +
                              "This argument is used with -fT option"))
    parser.add_argument("-featD",
                        "--feature-dir",
                        help="The output directory for the computed mfccs")
    # training
    parser.add_argument("-turnn",
                        "--train-unirnn",
                        help="Train the uni-directional RNN",
                        action="store_true")
    parser.add_argument("-tOD",
                        "--train-output-dir",
                        help="The training output directory")
    # validation
    parser.add_argument("-vM",
                        "--validate-model",
                        help="Running validation for validation audio",
                        action="store_true")
    parser.add_argument("-mID",
                        "--model-input-dir",
                        help="Choose the model you want to use to validate")
    parser.add_argument("-vAP",
                        "--validation-audio-path",
                        help="The validation audio path")
    parser.add_argument("-vPP",
                        "--validation-phoneme-path",
                        help="The validation phoneme path")
    return parser.parse_args()

def run_ctc():
    """
    Main function for running the program
    """
    args = parse_args()
    # for the feature transforming
    if args.feature_transform:
        timit_dir = args.timit_dir
        feature_dir = args.feature_dir
        if not os.path.exists(timit_dir):
            print("TIMIT directory doesnot exist!")
            sys.exit(0)
        if not os.path.exists(feature_dir):
            os.makedirs(feature_dir)
        featpickle_path = os.path.join(feature_dir, "features.pickle")
        featext.compute_and_store_features(timit_dir,
                                           featpickle_path)
    # for training
    elif args.train_unirnn:
        outputdir = args.train_output_dir
        featdir = args.feature_dir
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
        featpickle_file = os.path.join(featdir, "features.pickle")
        if not os.path.exists(featpickle_file):
            print("features not exist. Please run feature transform first")
            sys.exit(0)
        model_path = os.path.join(outputdir, "inference_model")
        train.train(model_path, featpickle_file)
    # for validation
    else:
        model_dir = args.model_input_dir
        model_path = os.path.join(model_dir, "inference_model.meta")
        audio_path = args.validation_audio_path
        phoneme_path = args.validation_phoneme_path
        if not os.path.isfile(audio_path):
            print("audio file not exist!")
            sys.exit(0)
        if not os.path.isfile(phoneme_path):
            print("phoneme file not exist!")
        if not os.path.isfile(model_path):
            print("model not exist! please train first")
        train.validate(audio_path, phoneme_path, model_path)

if __name__ == "__main__":
    run_ctc()
