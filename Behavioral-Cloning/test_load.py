import numpy as np
import os

array = np.arange(5)
np.save('geekfile.npy', array)

loaded = np.load('geekfile.npy', encoding='latin1')

print('loaded: ', loaded)
print('original: ', array)


print('os.listdir(./data/): ', os.listdir('./data/'))

shards = [x for x in os.listdir('./data/') if x.endswith('.npy')]

print(shards)

for shard in shards:
    shard_path = os.path.join('./data/', shard)
    print(shard_path)
    print(np.load(shard_path, allow_pickle = True, encoding='latin1'))
    # with open(shard_path, 'rb') as f:
    # 	data = np.load(f)