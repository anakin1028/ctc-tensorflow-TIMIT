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
        "iy": 0,
        "ih": 1,
        "eh": 2,
        "ey": 3,
        "ae": 4,
        "aa": 5,
        "aw": 6,
        "ay": 7,
        "ah": 8,
        "ao": 9,
        "oy": 10,
        "ow": 11,
        "uh": 12,
        "uw": 13,
        "ux": 14,
        "er": 15,
        "ax": 16,
        "ix": 17,
        "axr": 18,
        "ax-h": 19,
        "jh": 20,
        "ch": 21,
        "b": 22,
        "d": 23,
        "g": 24,
        "p": 25,
        "t": 26,
        "k": 27,
        "dx": 28,
        "s": 29,
        "sh": 30,
        "z": 31,
        "zh": 32,
        "f": 33,
        "th": 34,
        "v": 35,
        "dh": 36,
        "m": 37,
        "n": 38,
        "ng": 39,
        "em": 40,
        "nx": 41,
        "en": 42,
        "eng": 43,
        "l": 44,
        "r": 45,
        "w": 46,
        "y": 47,
        "hh": 48,
        "hv": 49,
        "el": 50,
        "bcl": 51,
        "dcl": 52,
        "gcl": 53,
        "pcl": 54,
        "tcl": 55,
        "kcl": 56,
        "q": 57,
        "pau": 58,
        "epi": 59,
        "h#": 60
    }
# map index to phoneme
IDX2PHONE = {PHONE_IDX_DICT[x]:x for x in PHONE_IDX_DICT}

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
                phone = line.strip().split(" ")[2]
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
                # store the features for si and sx
                if basename[:2] == 'si' or basename[:2]=='sx':
                    print(basename)
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
