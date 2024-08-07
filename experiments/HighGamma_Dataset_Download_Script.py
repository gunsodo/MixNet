import numpy as np
from utility import butter_bandpass_filter, Schirrmeister2017_our_own_path
from os.path import expanduser
home = expanduser("~")
import mne
import moabb
from moabb.paradigms import LeftRightImagery, MotorImagery

PATH = 'raw/'
for directory in [PATH]:
    if not os.path.exists(directory):
        os.makedirs(directory)

n_subjs = 14
num_class = 4
def read_raw(subject, num_class):
    dataset = Schirrmeister2017_our_own_path()
    paradigm = MotorImagery(n_classes=num_class)
    X, y, meta = paradigm.get_data(dataset=dataset, subjects=[subject])
    tr_id = list(meta[meta['run']=='train'].index)
    te_id = list(meta[meta['run']=='test'].index)
    X_tr, X_te = X[tr_id], X[te_id]
    y_tr, y_te = y[tr_id], y[te_id]
    return X_tr, y_tr, X_te, y_te

def save_rawdata(PATH, NAME, X_train, y_train, X_test, y_test):
    np.save(PATH+'X_train_'+NAME+'.npy', X_train)
    np.save(PATH+'X_test_'+NAME+'.npy', X_test)
    np.save(PATH+'y_train_'+NAME+'.npy', y_train)
    np.save(PATH+'y_test_'+NAME+'.npy', y_test)
    print('Save Done for : ', NAME)
    
for s in range(1, n_subjs+1):
    X_tr, y_tr, X_eval, y_eval = read_raw(s, num_class)
    print("Checking overall dimension Data Tr {} and Te {}".format(X_tr.shape, X_eval.shape))
    print("Checking overall dimension Label Tr {} and Te {}".format(y_tr.shape, y_eval.shape))
    
    SAVE_NAME = 'S{:02d}'.format(s)
    save_rawdata(PATH, SAVE_NAME, X_tr, y_tr, X_eval, y_eval)
