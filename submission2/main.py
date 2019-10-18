from sklearn.metrics import precision_recall_curve
import numpy as np

from dataloader import get_audio_loader
from solver import Solver

# Set paths
PATH = '../data'
DATA_PATH = f'{PATH}/mediaeval-2019-jamendo/'
LABELS_TXT = f'{PATH}/moodtheme_split.txt'
TRAIN_PATH = f'{PATH}/autotagging_moodtheme-train.tsv'
VAL_PATH = f'{PATH}/autotagging_moodtheme-validation.tsv'
TEST_PATH = f'{PATH}/autotagging_moodtheme-test.tsv'

CONFIG = {
        'log_dir': './output',
        'batch_size': 8
    }

def get_labels_to_idx(labels_txt):
    labels_to_idx = {}
    tag_list = []
    with open(labels_txt) as f:
        lines = f.readlines()

    for i,l in enumerate(lines):
        tag_list.append(l.strip())
        labels_to_idx[l.strip()] = i

    return labels_to_idx, tag_list

def train():
    config = CONFIG
    labels_to_idx, tag_list = get_labels_to_idx(LABELS_TXT)    

    train_loader1 = get_audio_loader(DATA_PATH, TRAIN_PATH, labels_to_idx, batch_size=config['batch_size'])
    train_loader2 = get_audio_loader(DATA_PATH, TRAIN_PATH, labels_to_idx, batch_size=config['batch_size'])
    val_loader = get_audio_loader(DATA_PATH, VAL_PATH, labels_to_idx, batch_size=config['batch_size'], shuffle=False, drop_last=False)
    solver = Solver(train_loader1,train_loader2, val_loader, tag_list, config)
    solver.train()

def predict():
    config = CONFIG
    labels_to_idx, tag_list = get_labels_to_idx(LABELS_TXT)

    test_loader = get_audio_loader(DATA_PATH, TEST_PATH, labels_to_idx, batch_size=config['batch_size'], shuffle=False, drop_last=False)

    solver = Solver(test_loader,None, None, tag_list, config)
    predictions = solver.test()

    np.save(f"{CONFIG['log_dir']}/predictions.npy", predictions)


if __name__=="__main__":

    #Train the data
    train()

    #Predict and create submissions
    predict()