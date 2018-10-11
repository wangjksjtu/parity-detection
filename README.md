# parity-detection

## Usage
    python train.py -h
    
    usage: train.py [-h] [--kbit KBIT] [--split_ratio SPLIT_RATIO]
                    [--batch_size BATCH_SIZE] [--num_epoch NUM_EPOCH]
                    [--learning_rate LEARNING_RATE] [--decay_rate DECAY_RATE]
                    [--log_dir LOG_DIR]

## Output
__python train.py --kbit 10__

    Epoch 00000: val_acc improved from -inf to 0.60651, saving model to logs/kbit_10/weights.hdf5
    0s - loss: 0.7065 - acc: 0.5292 - val_loss: 0.6764 - val_acc: 0.6065
    Epoch 2/30
    Epoch 00001: val_acc improved from 0.60651 to 0.66568, saving model to logs/kbit_10/weights.hdf5
    0s - loss: 0.6552 - acc: 0.6079 - val_loss: 0.6308 - val_acc: 0.6657
    Epoch 3/30
    Epoch 00002: val_acc improved from 0.66568 to 0.75148, saving model to logs/kbit_10/weights.hdf5
    0s - loss: 0.6088 - acc: 0.7012 - val_loss: 0.5864 - val_acc: 0.7515
    Epoch 4/30
    Epoch 00003: val_acc improved from 0.75148 to 0.85207, saving model to logs/kbit_10/weights.hdf5
    0s - loss: 0.5619 - acc: 0.8047 - val_loss: 0.5387 - val_acc: 0.8521
    Epoch 5/30
    Epoch 00004: val_acc improved from 0.85207 to 0.94379, saving model to logs/kbit_10/weights.hdf5
    0s - loss: 0.5128 - acc: 0.8950 - val_loss: 0.4889 - val_acc: 0.9438
    Epoch 6/30
    Epoch 00005: val_acc improved from 0.94379 to 0.96746, saving model to logs/kbit_10/weights.hdf5
    0s - loss: 0.4622 - acc: 0.9548 - val_loss: 0.4391 - val_acc: 0.9675
    Epoch 7/30
    Epoch 00006: val_acc improved from 0.96746 to 0.98521, saving model to logs/kbit_10/weights.hdf5
    0s - loss: 0.4116 - acc: 0.9883 - val_loss: 0.3910 - val_acc: 0.9852
    Epoch 8/30
    Epoch 00007: val_acc improved from 0.98521 to 0.99408, saving model to logs/kbit_10/weights.hdf5
    0s - loss: 0.3639 - acc: 0.9942 - val_loss: 0.3459 - val_acc: 0.9941
    Epoch 9/30
    Epoch 00008: val_acc improved from 0.99408 to 0.99704, saving model to logs/kbit_10/weights.hdf5
    0s - loss: 0.3197 - acc: 0.9985 - val_loss: 0.3042 - val_acc: 0.9970
    Epoch 10/30
    Epoch 00009: val_acc did not improve
    0s - loss: 0.2794 - acc: 1.0000 - val_loss: 0.2666 - val_acc: 0.9970
    Epoch 11/30
    Epoch 00010: val_acc improved from 0.99704 to 1.00000, saving model to logs/kbit_10/weights.hdf5
    0s - loss: 0.2437 - acc: 1.0000 - val_loss: 0.2335 - val_acc: 1.0000

__python train.py --kbit 11__

    Epoch 00000: val_acc improved from -inf to 0.50444, saving model to logs/kbit_11/weights.hdf5
    0s - loss: 0.7221 - acc: 0.4402 - val_loss: 0.6960 - val_acc: 0.5044
    Epoch 2/30
    Epoch 00001: val_acc improved from 0.50444 to 0.66124, saving model to logs/kbit_11/weights.hdf5
    0s - loss: 0.6686 - acc: 0.5918 - val_loss: 0.6459 - val_acc: 0.6612
    Epoch 3/30
    Epoch 00002: val_acc improved from 0.66124 to 0.77811, saving model to logs/kbit_11/weights.hdf5
    0s - loss: 0.6165 - acc: 0.7332 - val_loss: 0.5895 - val_acc: 0.7781
    Epoch 4/30
    Epoch 00003: val_acc improved from 0.77811 to 0.90533, saving model to logs/kbit_11/weights.hdf5
    0s - loss: 0.5552 - acc: 0.8528 - val_loss: 0.5217 - val_acc: 0.9053
    Epoch 5/30
    Epoch 00004: val_acc improved from 0.90533 to 0.96893, saving model to logs/kbit_11/weights.hdf5
    0s - loss: 0.4829 - acc: 0.9308 - val_loss: 0.4432 - val_acc: 0.9689
    Epoch 6/30
    Epoch 00005: val_acc improved from 0.96893 to 0.99112, saving model to logs/kbit_11/weights.hdf5
    0s - loss: 0.4023 - acc: 0.9789 - val_loss: 0.3594 - val_acc: 0.9911
    Epoch 7/30
    Epoch 00006: val_acc improved from 0.99112 to 1.00000, saving model to logs/kbit_11/weights.hdf5
    0s - loss: 0.3212 - acc: 0.9993 - val_loss: 0.2803 - val_acc: 1.0000

__python train.py --kbit 12__

    Epoch 00000: val_acc improved from -inf to 0.77515, saving model to logs/kbit_12/weights.hdf5
    1s - loss: 0.6387 - acc: 0.6454 - val_loss: 0.5397 - val_acc: 0.7751
    Epoch 2/30
    Epoch 00001: val_acc improved from 0.77515 to 0.92899, saving model to logs/kbit_12/weights.hdf5
    0s - loss: 0.4559 - acc: 0.8451 - val_loss: 0.3684 - val_acc: 0.9290
    Epoch 3/30
    Epoch 00002: val_acc improved from 0.92899 to 0.99778, saving model to logs/kbit_12/weights.hdf5
    0s - loss: 0.2902 - acc: 0.9730 - val_loss: 0.2197 - val_acc: 0.9978
    Epoch 4/30
    Epoch 00003: val_acc improved from 0.99778 to 1.00000, saving model to logs/kbit_12/weights.hdf5
    0s - loss: 0.1702 - acc: 0.9978 - val_loss: 0.1294 - val_acc: 1.0000


__python train.py --kbit 13__

    Epoch 00000: val_acc improved from -inf to 0.90126, saving model to logs/kbit_13/weights.hdf5
    1s - loss: 0.5924 - acc: 0.7077 - val_loss: 0.4413 - val_acc: 0.9013
    Epoch 2/30
    Epoch 00001: val_acc improved from 0.90126 to 0.99963, saving model to logs/kbit_13/weights.hdf5
    0s - loss: 0.2916 - acc: 0.9750 - val_loss: 0.1678 - val_acc: 0.9996
    Epoch 3/30
    Epoch 00002: val_acc improved from 0.99963 to 1.00000, saving model to logs/kbit_13/weights.hdf5
    0s - loss: 0.1089 - acc: 1.0000 - val_loss: 0.0691 - val_acc: 1.0000
