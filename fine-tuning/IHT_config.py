# Author: Zeheng Bai
##### INHERIT DNABERT MODELS #####
from basicsetting import *

### You should replace '.' with their paths if they do not in $Your_current_workpath/INHERIT/INHERIT ###
### Please put the pre-trained models in the same path of this code. If not, please use their absolute paths ###
CONFIG_PATH = './bert-config-6-conditional'
BAC_TR_PATH = $BAC_TR_PATH
PHA_TR_PATH = $PHA_TR_PATH
BAC_VAL_PATH = $BAC_VAL_PATH
PHA_VAL_PATH = $PHA_VAL_PATH
PATIENCE = 3
KMERS = 6
SEGMENT_LENGTH = 500
TR_BATCHSIZE = 64
VAL_BATCHSIZE = 32
TR_WORKERS = 3
VAL_WORKERS = 3
EPOCHS = 100
THRESHOLD = 0.5
LEARNING_RATE = $LR