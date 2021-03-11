import numpy as np
from tqdm import tqdm
import os
from navier_stocks import navier_stokes_2d
from heat_equatation import heat_2d
import argparse


def generate_dataset(dataset, num_samples, path, params):
    if dataset == 'navier_stocks':
        generate_sample = navier_stokes_2d
    elif dataset == 'heat':
        generate_sample = heat_2d
    else:
        raise ValueError(f'Unknown dataset: {dataset}')

    os.mkdir(os.path.join(path, dataset))
    l = len(str(num_samples))

    for i in tqdm(range(num_samples)):
        input, solution = generate_sample(i, **params)
        np.save(f"input_{str(i).rjust(l, '0')}", input)
        np.save(f"solution_{str(i).rjust(l, '0')}", solution)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['navier_stocks', 'heat'], default='navier_stocks')
    parser.add_argument('--num_samples', type=int, default=1200)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--path', type=str, default='../datasets', help='path to folder with datasets')
    parser.add_argument('--s', type=int, default=256, help='spatial resolution')
    parser.add_argument('--T', type=float, default=50.0, help='end time')
    parser.add_argument('--num_steps', type=int, default=50, help='number of time points in which solution computed')
    args = parser.parse_args()
    generate_dataset(args.dataset, args.num_samples, args.path, vars(args))


if __name__ == '__main__':
    main()
