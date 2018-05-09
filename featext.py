"""
Module to extract the features from audio and parse the phoneme streams
"""
import librosa
import pickle
import numpy as np
import os

# to be refactored
TIMIT_PATH="./timit"
TRAIN_PATH=os.path.join(TIMIT_PATH, 'train')
FEAT_PICKFILE=os.path.join("./feature_input", "features.pickle")
# 61 + NULL
NUM_PHONEME_CLASSES = 62
PHONE_IDX_DICT = {
        "iy": 1,
        "ih": 2,
        "eh": 3,
        "ey": 4,
        "ae": 5,
        "aa": 6,
        "aw": 7,
        "ay": 8,
        "ah": 9,
        "ao": 10,
        "oy": 11,
        "ow": 12,
        "uh": 13,
        "uw": 14,
        "ux": 15,
        "er": 16,
        "ax": 17,
        "ix": 18,
        "axr": 19,
        "ax-h": 20,
        "jh": 21,
        "ch": 22,
        "b": 23,
        "d": 24,
        "g": 25,
        "p": 26,
        "t": 27,
        "k": 28,
        "dx": 29,
        "s": 30,
        "sh": 31,
        "z": 32,
        "zh": 33,
        "f": 34,
        "th": 35,
        "v": 36,
        "dh": 37,
        "m": 38,
        "n": 39,
        "ng": 40,
        "em": 41,
        "nx": 42,
        "en": 43,
        "eng": 44,
        "l": 45,
        "r": 46,
        "w": 47,
        "y": 48,
        "hh": 49,
        "hv": 50,
        "el": 51,
        "bcl": 52,
        "dcl": 53,
        "gcl": 54,
        "pcl": 55,
        "tcl": 56,
        "kcl": 57,
        "q": 58,
        "pau": 59,
        "epi": 60,
        "h#": 61
    }
    
class Phonemefeat:
    """
    Class defines the input features
    """
    def __init__(self, features, phnfile, wrdfile):
        """
        Function used to construct the object
        """
        self.features = features
        self.phnfile = phnfile
        self.wrdfile = wrdfile

    def get_phone_stream(self):
        """
        Parse the phoneme file in TIMIT
        The file looks like:
        0 3050 h#
        3050 4559 sh
        4559 5723 ix
        5723 6642 hv
        6642 8772 eh
        8772 9190 dcl
        9190 10337 jh
        10337 11517 ih
        11517 12500 dcl
        12500 12640 d
        44586 46720 h#
        """
        phones = []
        with open(self.phnfile, 'r') as fh:
            for line in fh:
                phone = line.strip.split(" ")[2]
                phones.append(PHONE_IDX_DICT[phone])
        return np.array(phones)

def mfcc_extract(audio_path):
   """
   :param audio_path: input file of audio
   :return 13 dimension of mfccs
   """

   y, sr = librosa.load(audio_path)
   mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
   return mfccs

def get_audio_feature(audio_path):
    """
    Get 26 mfcc features : 1st order + 2nd order
    :param audio_path: input file path
    :return a shape with (timestamps * 26) array each row contains 26 mfccs
     to represent the particular timestamp
    """

    mfccs = mfcc_extract(audio_path)
    deltas = librosa.feature.delta(mfccs)
    features = np.hstack((mfccs.T, deltas.T))
    return features

def compute_and_store_training_features(train_path):
    """
    Get all the training phoneme datas
    """
    train_datas = []
    for dirpath, dirnames, filenames in os.walk(train_path):
        # check all the files
        for filename in filenames:
            filenamesplits = os.path.splitext(filename)
            basename = filenamesplits[0]
            extname = filenamesplits[1]
            if extname == '.wav':
                fullpath = os.path.join(dirpath, filename)
                features = get_audio_feature(fullpath)
                phnfile = os.path.join(dirpath, basename+'.phn')
                wrdfile = os.path.join(dirpath, basename+'.wrd')
                train_datas.append(Phonemefeat(features, phnfile, wrdfile))
    with open(FEAT_PICKFILE, 'wb') as fh:
        pickle.dump(train_datas, fh)

def get_train_data():
    with open(FEAT_PICKFILE, 'rb') as fh:
        features = pickle.load(fh)
    return features

if __name__ == "__main__":
    #compute_and_store_training_features(TRAIN_PATH)
    xx = get_train_data()
