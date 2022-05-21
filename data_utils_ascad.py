import numpy as np
import tensorflow as tf
import h5py

import os, sys

class Dataset:
    def __init__(self, data_path, split):
        self.data_path = data_path
        self.split = split

        corpus = h5py.File(data_path, 'r')
        if split == 'train':
            split_key = 'Profiling_traces'
        elif split == 'test':
            split_key = 'Attack_traces'

        self.traces = corpus[split_key]['traces'][()]
        self.labels = np.reshape(corpus[split_key]['labels'][()], [-1, 1])
        self.num_samples = self.traces.shape[0]

        self.traces = self.traces.astype(np.float32)

        self.plaintexts = self.GetPlaintexts(corpus[split_key]['metadata'])
        self.masks = self.GetMasks(corpus[split_key]['metadata'])
        self.keys = self.GetKeys(corpus[split_key]['metadata'])

    
    def GetPlaintexts(self, metadata):
        plaintexts = []
        for i in range(len(metadata)):
            plaintexts.append(metadata[i]['plaintext'][2])
        return np.array(plaintexts)


    def GetKeys(self, metadata):
        keys = []
        for i in range(len(metadata)):
            keys.append(metadata[i]['key'][2])
        return np.array(keys)


    def GetMasks(self, metadata):
        masks = []
        for i in range(len(metadata)):
            masks.append(np.array(metadata[i]['masks']))
        masks = np.stack(masks, axis=0)
        return masks


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

    dataset = Dataset(data_path, batch_size, split)

    print("traces    : "+str(dataset.traces.shape))
    print("labels    : "+str(dataset.labels.shape))
    print("plaintext : "+str(dataset.plaintexts.shape))
    print("keys      : "+str(dataset.keys.shape))
    print("")
    print("")

    tfrecords = dataset.GetTFRecords()
    iterator = iter(tfrecords)
    for i in range(1):
        tr, lbl = iterator.get_next()
        print(str(tr.shape)+' '+str(lbl.shape))
        print(str(tr[:5, :5]))
        print(str(lbl[:5, :]))
        print("")

