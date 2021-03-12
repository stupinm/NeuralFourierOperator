import torch
import torch.utils.data as torch_data
from torchvision import transforms
import numpy as np
import os


class ToTensor(object):
    def __call__(self, sample):
        return list(map(torch.from_numpy, sample))


class Downsample(object):
    def __init__(self, s, t):
        self.s = s
        self.t = t

    def __call__(self, sample):
        input, label = sample
        s, t = self.s, self.t
        return input[::s, ::s, ::self.t], label[::s, ::s, ::t]


class PadCoordinates(object):
    def __init__(self, S):
        self.S = S

    def __call__(self, sample):
        S = self.S
        input, label = sample
        gridx = torch.tensor(np.linspace(0, 1, S), dtype=torch.float32).reshape(S, 1, 1).repeat([1, S, 1])
        gridy = torch.tensor(np.linspace(0, 1, S), dtype=torch.float32).reshape(1, S, 1).repeat([S, 1, 1])
        input = torch.cat((gridx.repeat([1, 1, 1]), gridy.repeat([1, 1, 1]), input), dim=-1)

        return input, label


class PDEDataset(torch_data.Dataset):
    def __init__(self, path, ids, l, ratio=0.2, transform=None):
        super(PDEDataset, self).__init__()
        self.path = path
        self.ids = ids
        self.transform = transform
        self.l = l
        self.ratio = ratio

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        solution = np.load(os.path.join(self.path, f"solution_{str(self.ids[idx]).rjust(self.l, '0')}.npy"))
        input_time_len = int(self.ratio * solution.shape[-1])
        input, label = solution[..., :input_time_len], solution[..., input_time_len:]
        if self.transform is not None:
            input, label = self.transform((input, label))

        return input.float(), label.float()


class Data(object):
    def __init__(self, args):
        self.path = os.path.join(args['datasets'], args['dataset'])
        self.train_val_ratio = args['train_val_ratio']
        self.num_samples = args['num_samples']
        self.batch_size = args['batch_size']
        self.s = args['s']
        self.t = args['t']
        self.ratio = args['input_output_ratio']
        self.S = args['S']

    def inspect_folder(self):
        npy_files_length = list(map(len, filter(lambda x: str.endswith(x, '.npy'), os.listdir(self.path))))
        num_files = len(npy_files_length) // 2
        npy_files_length = set(npy_files_length)
        assert len(npy_files_length) == 2
        l1, l2 = min(npy_files_length), max(npy_files_length)
        l1, l2 = l1 - len(''.join(['input_', '.npy'])), l2 - len(''.join(['solution_', '.npy']))
        assert l1 == l2
        l = l1

        if self.num_samples > num_files:
            print(f'Not enough samples in {self.path}, num_samples was decreased to: {num_files}')
            self.num_samples = num_files

        assert self.num_samples >= self.batch_size
        if self.num_samples % self.batch_size != 0:
            print(f'Number of samples non multiple to batch_size, skip last non-full batch')
            self.num_samples -= self.num_samples % self.batch_size

        return l

    def get_transforms(self):
        transforms_train = transforms.Compose([
            Downsample(self.s, self.t),
            ToTensor(),
            PadCoordinates(self.S)
        ])
        transforms_val = transforms.Compose([
            Downsample(self.s, self.t),
            ToTensor(),
            PadCoordinates(self.S)
        ])
        return transforms_train, transforms_val

    def get_dataloaders(self):
        l = self.inspect_folder()
        permutation = np.random.permutation(self.num_samples)
        train_len = int(self.num_samples * self.train_val_ratio)
        train_len = train_len - train_len % self.batch_size
        train_ids, val_ids = permutation[:train_len], permutation[train_len:]

        transforms_train, transforms_val = self.get_transforms()
        train_dataset = PDEDataset(self.path, train_ids, l, self.ratio, transforms_train)
        val_dataset = PDEDataset(self.path, val_ids, l, self.ratio, transforms_val)

        train_dataloader = torch_data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_dataloader = torch_data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        return train_dataloader, val_dataloader
