import torch
from easydict import EasyDict as edict

config = edict()

########
# MAIN #
########
# main is the config section related to basic info on the project
# data repo, data format, folding etc... data preparation
config.main = edict()
config.main.PROJECT_PATH = "task/tweet_disaster/"
config.main.TRAIN_FILE = "data/tweet_disaster/train.csv"
config.main.TEST_FILE = "data/tweet_disaster/test.csv"
config.main.FOLD_FILE = "data/tweet_disaster/train_folds.csv"
config.main.FOLD_METHOD = "SKF"
config.main.TARGET_VAR = "target"
config.main.TEXT_VAR = "text"
config.main.DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
config.main.N_CLASS = 2
config.main.MAX_LEN = 160

###################
# HYPERPARAMETERS #
###################
config.hyper = edict()
config.hyper.TRAIN_BS = 64                                                                              #Batch size for training pass
config.hyper.VALID_BS = 32                                                                               #Batch size for validation pass
config.hyper.EPOCHS = 5                                                                                 #Number of epochs
config.hyper.LR = 1e-4                                                                                  #Learning rate