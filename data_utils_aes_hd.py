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
        
        self.traces = np.load(os.path.join(data_path, split_key + '_traces_AES_HD.npy'))
        self.labels = np.load(os.path.join(data_path, split_key + '_labels_AES_HD.npy'))
        self.plaintexts = np.load(os.path.join(data_path, split_key + '_ciphertext_AES_HD.npy'))

        self.num_samples = self.traces.shape[0]
        self.keys = np.zeros((self.num_samples,), dtype=int)

        self.traces = self.traces.astype(np.float32)
        self.labels = self.GetBitLabels(self.labels.astype(np.int32))
        self.plaintexts = self.plaintexts.astype(np.int32)

    
    def GetTFRecords(self, batch_size, training=False):
        dataset = tf.data.Dataset.from_tensor_slices((self.traces, self.labels))
        if training == True:
            return dataset.repeat().shuffle(self.traces.shape[0]).batch(batch_size, drop_remainder=True)
        else:
            return dataset.batch(batch_size, drop_remainder=True)


    def GetDataset(self):
        return self.traces, self.labels


    def GetBitLabels(self, labels):
        nsamples = labels.shape[0]

        nclasses = 8

        bit_labels = np.zeros((nsamples, nclasses), dtype=np.float32)
        for i in range(nsamples):
            cur_label = labels[i]
            for b in range(nclasses):
                temp = np.right_shift(cur_label, b)
                bit_labels[i, b] = float(temp & 1)

        return bit_labels



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

