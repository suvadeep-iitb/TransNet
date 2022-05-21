import numpy as np
import tensorflow as tf
import h5py

import os, sys

class Dataset:
    def __init__(self, data_path, split):
        self.data_path = data_path
        self.split = split

        if split == 'train':
            split_key = 'profiling'
        elif split == 'test':
            split_key = 'attack'
        
        self.traces = np.load(os.path.join(data_path, split_key + '_traces_AES_RD.npy'))
        self.labels = np.load(os.path.join(data_path, split_key + '_labels_AES_RD.npy'))
        self.plaintexts = np.load(os.path.join(data_path, split_key + '_plaintext_AES_RD.npy'))

        self.num_samples = self.traces.shape[0]

        key = np.load(os.path.join(data_path, 'key.npy'))[0]
        self.keys = np.zeros((self.num_samples,), dtype=int) + key

        self.traces = self.traces.astype(np.float32)
        self.labels = self.labels.astype(np.int32)
        self.plaintexts = self.plaintexts.astype(np.int32)[:, 0]

        self.labels = np.reshape(self.labels, [-1, 1])

    
    def GetTFRecords(self, batch_size, training=False):
        dataset = tf.data.Dataset.from_tensor_slices((self.traces, self.labels))
        if training == True:
            return dataset.repeat().shuffle(self.traces.shape[0]).batch(batch_size, drop_remainder=True)
        else:
            return dataset.batch(batch_size, drop_remainder=True)


    def GetDataset(self):
        return self.traces, self.labels


if __name__ == '__main__':
    data_path = sys.argv[1]
    batch_size = int(sys.argv[2])
    split = sys.argv[3]

    dataset = Dataset(data_path, split)

    print("traces    : "+str(dataset.traces.shape))
    print("labels    : "+str(dataset.labels.shape))
    print("plaintext : "+str(dataset.plaintexts.shape))
    print("keys      : "+str(dataset.keys.shape))
    print("")
    print("")

    tfrecords = dataset.GetTFRecords(batch_size)
    iterator = iter(tfrecords)
    for i in range(1):
        tr, lbl = iterator.get_next()
        print(str(tr.shape)+' '+str(lbl.shape))
        print(str(tr[:5, :5]))
        print(str(lbl[:5, :]))
        print("")

