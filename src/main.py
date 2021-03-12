import torch
from neural_fourier import FourierNet
from train import Trainer
from data import Data
from utils import parse_args, dump_config, mkdirs
import sys
from contextlib import redirect_stdout
import os

def main():
    config = sys.argv[1]
    args = parse_args(config)

    mkdirs(args)
    dump_config(config, args)

    net = FourierNet(args['n_layers'], args['n_modes'], args['width']).to(args['device'])
    optimizer = torch.optim.Adam(net.parameters(), lr=args['learning_rate'], weight_decay=args['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args['scheduler_step'], gamma=args['scheduler_gamma'])

    with open(os.path.join(args['experiments'], args['exp_name'], 'log.txt'), 'w') as f:
        with redirect_stdout(f):
            data = Data(args)
            train_loader, val_loader = data.get_dataloaders()

            trainer = Trainer(args, net, optimizer, scheduler, train_loader, val_loader)
            trainer.train()


if __name__ == '__main__':
    main()
