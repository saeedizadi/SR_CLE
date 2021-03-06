import argparse
from copy import deepcopy


### note: argparse Namespace works like a dictiornary. You can treat as a dictionary -> define some arguments (keys) and assign the default values ###


# --- define the main namespace for arguments
defaults = argparse.Namespace()
defaults.VISDOM_PORT = 8100
defaults.CUDA = False


# --- define the namespace for defaults for train mode
default_train = argparse.Namespace()
default_train.BATCH_SIZE = 128
default_train.LEARNING_RATE = 0.001
default_train.EPOCHS = 300
default_train.MOMENTUM = 0.9
default_train.WEIGHT_DECAY = 0.0005
default_train.PATCH_SIZE  = 64
default_train.IMAGE_SIZE  = 1024
default_train.DOWNSCALE_RATIO = 8
default_train.SR_TRAIN_DIR = '/local-scratch/saeedI/SR_CLE/data/highres/train'
default_train.LR_TRAIN_DIR = '/local-scratch/saeedI/SR_CLE/data/lowres/train'
default_train.SR_VAL_DIR = '/local-scratch/saeedI/SR_CLE/data/highres/val'
default_train.LR_VAL_DIR = '/local-scratch/saeedI/SR_CLE/data/lowres/val'
default_train.NUM_WORKER = 2
default_train.LOG_STEP= 10
default_train.STATE= 20 # for resume
default_train.SAVE_DIR= '../checkpoints'


default_test = argparse.Namespace()
default_test.BATCH_SIZE = 4
default_test.PATCH_SIZE  = 64
default_test.IMAGE_SIZE  = 1024
default_test.DOWNSCALE_RATIO= 8
default_test.SR_TEST_DIR = '/local-scratch/saeedI/SR_CLE/data/highres/test'
default_test.LR_TEST_DIR = '/local-scratch/saeedI/SR_CLE/data/lowres/test'
default_test.NUM_WORKER = 2
default_test.STATE= 70
default_test.SAVE_DIR= '../results'
default_test.WEIGHT_DIR= '../checkpoints'


default_evaluate = argparse.Namespace()
default_evaluate.RESULTS= '../results'
default_evaluate.HR_TEST_DIR= '../data/highres/test'
default_evaluate.METHODS= ['srdensenet', 'bicubic']


default_show = argparse.Namespace()
default_show.HIGHRES= '/local-scratch/saeedI/SR_CLE/data/highres/test'
default_show.LOWRES= '/local-scratch/saeedI/SR_CLE/data/lowres/test'
default_show.RESULTS= '../results'
default_show.METHODS = ['srdensenet', 'bicubic']


# --- define the function to define the parsers and subparsers and get the arguments
def get_arguments():
    parser = argparse.ArgumentParser(description='')

    # --- what comes after the calling the filename of the program
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('-visp','--visdom-port', type=int, default=defaults.VISDOM_PORT)



    # --- Stating that we want to have subparsers
    subparser = parser.add_subparsers(dest='mode')
    subparser.required = True

    # --- define subparsers --> train
    parser_train = subparser.add_parser('train')
    parser_train.add_argument('-nepoch', '--num-epochs', type=int, default=default_train.EPOCHS)
    parser_train.add_argument('-lr', '--learning-rate', type=float, default=default_train.LEARNING_RATE)
    parser_train.add_argument('-mom', '--momentum', type=float, default=default_train.MOMENTUM)
    parser_train.add_argument('--weight-decay', type=float, default=default_train.WEIGHT_DECAY)
    parser_train.add_argument('-psize', '--patch-size', type=int, default=default_train.PATCH_SIZE)
    parser_train.add_argument('--image-size', type=int, default=default_train.IMAGE_SIZE)
    parser_train.add_argument('-bsize', '--batch-size', type=int, default=default_train.BATCH_SIZE)
    parser_train.add_argument('-dscale', '--downscale-ratio', type=int, default=default_train.DOWNSCALE_RATIO)
    parser_train.add_argument('--srtraindir', type=str, default=default_train.SR_TRAIN_DIR)
    parser_train.add_argument('--lrtraindir', type=str, default=default_train.LR_TRAIN_DIR)
    parser_train.add_argument('--srvaldir', type=str, default=default_train.SR_VAL_DIR)
    parser_train.add_argument('--lrvaldir', type=str, default=default_train.LR_VAL_DIR)
    parser_train.add_argument('--savedir', type=str, default=default_train.SAVE_DIR)
    parser_train.add_argument('--log-step', type=int, default=default_train.LOG_STEP)
    parser_train.add_argument('--resume', action='store_true')
    parser_train.add_argument('--state', type=int, default=default_train.STATE)

    # --- define subparsers --> test
    parser_test = subparser.add_parser('test')
    parser_test.add_argument('-psize', '--patch-size', type=int, default=default_test.PATCH_SIZE)
    parser_test.add_argument('--image-size', type=int, default=default_test.IMAGE_SIZE)
    parser_test.add_argument('-bsize', '--batch-size', type=int, default=default_test.BATCH_SIZE)
    parser_test.add_argument('-dscale', '--downscale-ratio', type=int, default=default_test.DOWNSCALE_RATIO)
    parser_test.add_argument('--srtestdir', type=str, default=default_test.SR_TEST_DIR)
    parser_test.add_argument('--lrtestdir', type=str, default=default_test.LR_TEST_DIR)
    parser_test.add_argument('--savedir', type=str, default=default_test.SAVE_DIR)
    parser_test.add_argument('--weightdir', type=str, default=default_test.WEIGHT_DIR)
    parser_test.add_argument('--state', type=int, default=default_test.STATE)

    parser_evaluate = subparser.add_parser('evaluate')
    parser_evaluate.add_argument('--resdir', type=str, default=default_evaluate.RESULTS)
    parser_evaluate.add_argument('--hrdir', type=str, default=default_evaluate.HR_TEST_DIR)
    parser_evaluate.add_argument('--methods', nargs='+', default=default_evaluate.METHODS)


    parser_show = subparser.add_parser('show')
    parser_show.add_argument('--hrdir', type=str, default=default_show.HIGHRES)
    parser_show.add_argument('--lrdir', type=str, default=default_show.LOWRES)
    parser_show.add_argument('--resdir', type=str, default=default_show.RESULTS)
    parser_show.add_argument('--methods', nargs='+', default=default_show.METHODS)
    args = parser.parse_args()
    return args
