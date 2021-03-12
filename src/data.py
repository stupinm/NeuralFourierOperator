import torch
import torch.utils.data as torch_data
from torchvision import transforms
import numpy as np
import os


class ToTensor(object):
    def __call__(self, *args):
        return list(map(torch.from_numpy, args))


class Downsample(object):
    def __init__(self, s, num_steps):
        self.s = s
        self.num_steps = num_steps

    def __call__(self, input, solution):
        s = self.s
        return self.input[::s, ::s], self.solution[::s, ::s, ::self.num_steps]


class PDEDataset(torch_data.Dataset):
    def __init__(self, path, ids, l, transform=None):
        super(PDEDataset, self).__init__()
        self.path = path
        self.ids = ids
        self.transform = transform
        self.l = l

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        input = np.load(os.path.join(self.path, f"input_{str(self.ids[idx]).rjust(self.l, '0')}"))
        solution = np.load(os.path.join(self.path, f"solution_{str(self.ids[idx]).rjust(self.l, '0')}"))
        if self.transform is not None:
            input, solution = self.transform(input, solution)

        return input, solution


class Data(object):
    def __init__(self, args):
        self.path = args['path']
        self.train_val_ratio = args['train_val_ratio']
        self.num_samples = args['num_samples']
        self.batch_size = args['batch_size']
        self.s = args['s']
        self.num_steps = args['num_steps']


    def inspect_folder(self):
        npy_files_length = set(list(map(len, filter(str.endswith('.npy'), os.listdir(self.path)))))
        assert len(npy_files_length) == 2
        l1, l2 = min(npy_files_length), max(npy_files_length)
        l1, l2 = l1 - len(''.join(['input_', '.npy'])), l2 - len(''.join(['solution_', '.npy']))
        assert l1 == l2
        l = l1

        if self.num_samples > l:
            print(f'Not enough samples in {self.path}, num_samples was decreased to: {self.num_samples}')
            self.num_samples = l

        assert self.num_samples >= self.batch_size
        if self.num_samples % self.batch_size != 0:
            print(f'Number of samples non multiple to batch_size, skip last non-full batch')
            self.num_samples -= self.num_samples % self.batch_size

        return l

    def get_transforms(self):
        transforms_train = transforms.Compose([
            Downsample(self.s, self.num_steps),
            ToTensor()
        ])
        transforms_val = transforms.Compose([
            Downsample(self.s, self.num_steps),
            ToTensor()
        ])
        return transforms_train, transforms_val

    def get_dataloaders(self):
        l = self.inspect_folder()
        permutation = np.random.permutation(self.num_samples)
        train_len = int(self.num_samples * self.train_val_ratio)
        train_ids, val_ids = permutation[:train_len], permutation[train_len:]

        transforms_train, transforms_val = self.get_transforms()
        train_dataset = PDEDataset(self.path, train_ids, l, transforms_train)
        val_dataset = PDEDataset(self.path, val_ids, l, transforms_val)

        train_dataloader = torch_data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_dataloader = torch_data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        return train_dataloader, val_dataloader
