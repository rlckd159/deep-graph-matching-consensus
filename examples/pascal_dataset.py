import os.path as osp

import argparse
import torch
from torch_geometric.datasets import PascalVOCKeypoints as PascalVOC
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader

from dgmc.utils import ValidPairDataset
from dgmc.models import DGMC, SplineCNN

parser = argparse.ArgumentParser()
parser.add_argument('--isotropic', action='store_true')
parser.add_argument('--dim', type=int, default=256)
parser.add_argument('--rnd_dim', type=int, default=128)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--num_steps', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--test_samples', type=int, default=1000)
args = parser.parse_args()

pre_filter = lambda data: data.pos.size(0) > 0  # noqa
transform = T.Compose([
    T.Delaunay(),
    T.FaceToEdge(),
    T.Distance() if args.isotropic else T.Cartesian(),
])

train_datasets = []
test_datasets = []
path = osp.join('..', 'data', 'PascalVOC')
for category in PascalVOC.categories:
    dataset = PascalVOC(path, category, train=True, transform=transform,
                        pre_filter=pre_filter)
    train_datasets += [ValidPairDataset(dataset, dataset, sample=True)]
    dataset = PascalVOC(path, category, train=False, transform=transform,
                        pre_filter=pre_filter)
    test_datasets += [ValidPairDataset(dataset, dataset, sample=True)]
train_dataset = torch.utils.data.ConcatDataset(train_datasets)
train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True,
                          follow_batch=['x_s', 'x_t'])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
psi_1 = SplineCNN(dataset.num_node_features, args.dim,
                  dataset.num_edge_features, args.num_layers, cat=False,
                  dropout=0.5)
psi_2 = SplineCNN(args.rnd_dim, args.rnd_dim, dataset.num_edge_features,
                  args.num_layers, cat=True, dropout=0.0)
model = DGMC(psi_1, psi_2, num_steps=args.num_steps).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

