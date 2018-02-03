import argparse



### note: argparse Namespace works like a dictiornary. You can treat as a dictionary -> define some arguments (keys) and assign the default values ###


# --- define the main namespace for arguments
defaults = argparse.Namespace()
defaults.VISDOM_PORT = 8100
defaults.CUDA = False


# --- define the namespace for defaults for train mode
default_train = argparse.Namespace()
default_train.BATCH_SIZE = 64
default_train.LEARNING_RATE = 0.01
default_train.EPOCHS = 1000
default_train.MOMENTUM = 0.9
default_train.PATCH_SIZE  = 128
default_train.IMAGE_SIZE  = 1024
default_train.DOWNSCALE_RATIO= 4
default_train.SR_TRAIN_DIR = '/local-scratch/saeedI/CLE/data/highres/train'
default_train.LR_TRAIN_DIR = '/local-scratch/saeedI/CLE/data/lowres/train'
default_train.SR_VAL_DIR = '/local-scratch/saeedI/CLE/data/highres/val'
default_train.LR_VAL_DIR = '/local-scratch/saeedI/CLE/data/lowres/val'
default_train.NUM_WORKER = 2
default_train.LOG_STEP= 100
default_train.SAVE_DIR= '../checkpoints'




# --- define the function to define the parsers and subparsers and get the arguments
def get_arguments():
    parser = argparse.ArgumentParser(description='')

    # --- what comes after the calling the filename of the program
    parser.add_argument('--cuda', action='store_true', default=defaults.CUDA)
    parser.add_argument('-visp','--visdom-port', type=int, default=defaults.VISDOM_PORT)



    # --- Stating that we want to have subparsers
    subparser = parser.add_subparsers(dest='mode')
    subparser.required = True

    # --- define subparsers --> train
    parser_train = subparser.add_parser('train')
    parser_train.add_argument('-nepoch', '--num-epochs', type=int, default=default_train.EPOCHS)
    parser_train.add_argument('-lr', '--learning-rate', type=float, default=default_train.LEARNING_RATE)
    parser_train.add_argument('-mom', '--momentum', type=float, default=default_train.MOMENTUM)
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


    args = parser.parse_args()
    return args
