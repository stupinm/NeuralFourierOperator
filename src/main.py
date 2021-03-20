import torch
from neural_fourier import FourierNet
from neural_fourier_3d import FourierNet3d
from train import Trainer
from data import Data
from utils import parse_args, dump_config, mkdirs
import sys
from contextlib import redirect_stdout
import os


def main():
    config = sys.argv[1]
    args = parse_args(config)
    command = args['command']

    mkdirs(command, args)
    dump_config(command, config, args)

    if args['net_arch'] == '2d':
        net_class = FourierNet
    elif args['net_arch'] == '3d':
        net_class = FourierNet3d

    net = net_class(args['n_layers'], args['n_modes'], args['width'], args['t_in'], args['t_out']).to(args['device'])
    optimizer = torch.optim.Adam(net.parameters(), lr=args['learning_rate'], weight_decay=args['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args['scheduler_step'], gamma=args['scheduler_gamma'])

    data = Data(args)
    train_loader, val_loader, test_loader = data.get_dataloaders()

    trainer = Trainer(args, net, optimizer, scheduler, train_loader, val_loader, test_loader)

    with open(os.path.join(args['experiments'], args['exp_name'], f'log_{command}.txt'), 'w') as f:
        with redirect_stdout(f):
            if command == 'train':
                trainer.train()
            elif command == 'test':
                trainer.load_model()
                trainer.test(test_loader)
            elif command == 'predict':
                trainer.load_model()
                trainer.predict(test_loader)
            else:
                raise ValueError(f'Unknown command: {command}')


if __name__ == '__main__':
    main()
