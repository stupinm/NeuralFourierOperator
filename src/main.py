import torch
from neural_fourier import FourierNet2d, FourierNet3d
from train import Trainer
from data import Data
from utils import parse_args, dump_config, mkdirs
import sys
from contextlib import redirect_stdout
import os


def get_default_args():
    default_args = {
        "batch_size": 50,
        "n_epochs": 500,
        "weight_decay": 0.0001,
        "learning_rate": 0.0025,
        "scheduler_step": 100,
        "scheduler_gamma": 0.5,
        "n_layers": 4,
        "n_modes_1": 12,
        "n_modes_2": 12,
        "n_modes_3": 12,
        "width": 20,
        "predictive_mode": "multiple_step",
        "input_output_ratio": 0.2,
        "S": 64,
        "s": 1,
        "t": 1,
        "num_samples": 1200,
        "val_ratio": 0.1,
        "test_ratio": 0.2,
        "seed": 42,
        "device": "cuda",
        "experiments": "../experiments",
        "datasets": "../datasets"
    }

    return default_args


def main():
    config = sys.argv[1]
    args = parse_args(config)
    args = {**get_default_args(), **args}
    command = args['command']

    mkdirs(command, args)
    dump_config(command, args)

    if args['net_arch'] == '2d':
        net_class = FourierNet2d
        n_modes = (args['n_modes_1'], args['n_modes_2'])
    elif args['net_arch'] == '3d':
        net_class = FourierNet3d
        n_modes = (args['n_modes_1'], args['n_modes_2'], args['n_modes_3'])

    net = net_class(args['n_layers'], n_modes, args['width'], args['t_in'], args['t_out']).to(args['device'])
    optimizer = torch.optim.Adam(net.parameters(), lr=args['learning_rate'], weight_decay=args['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args['scheduler_step'], gamma=args['scheduler_gamma'])

    data = Data(args)
    train_loader, val_loader, test_loader = data.get_dataloaders()

    trainer = Trainer(args, net, optimizer, scheduler, train_loader, val_loader, test_loader)

    with open(os.path.join(args['experiments'], args['exp_name'], f'log_{command}.txt'), 'w') as f:
        with redirect_stdout(f):
            if command == 'train':
                trainer.train()
            elif command == 'train&test':
                loss = trainer.test(test_loader).item()
                print(f'Test loss: {loss / len(test_loader)}')
            elif command == 'test':
                trainer.load_model()
                loss = trainer.test(test_loader).item()
                print(f'Test loss: {loss / len(test_loader)}')
            elif command == 'predict':
                trainer.load_model()
                trainer.predict(test_loader)
            else:
                raise ValueError(f'Unknown command: {command}')


if __name__ == '__main__':
    main()
