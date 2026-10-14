import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from prior.tabiclv2.dataloader import get_tabiclv2_dataset, TabICLv2DiskDataLoader

dataloader = TabICLv2DiskDataLoader(data_dir = '/Users/code/mt_tree/dataset', batch_size=4, shuffle=True)

for i, (X, y, d, seq_lens, train_sizes) in enumerate(dataloader):
    # print()
    for tensor_X, tensor_y in zip(X, y):
        print(tensor_X.shape, tensor_y.shape)
    print(type(X), type(y), d, seq_lens, train_sizes)
    break

