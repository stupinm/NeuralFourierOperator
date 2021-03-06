import json
from shutil import copy
import os


def parse_args(file):
    with open(file, 'r') as f:
        args = json.load(f)

    return args

def mkdirs(args):
    exp_folder = os.path.join(args['experiments'], args['exp_name'])
    os.mkdir(exp_folder)
    os.mkdir(os.path.join(exp_folder, 'tensorboard'))


def dump_config(config, args):
    copy(config, os.path.join(args['experiments'], args['exp_name'], 'config.json'))
