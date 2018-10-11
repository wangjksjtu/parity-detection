# parity-detection

## Usage
    python train.py -h

    usage: train.py [-h] [--kbit KBIT] [--mask MASK] [--split_ratio SPLIT_RATIO]
                    [--batch_size BATCH_SIZE] [--num_epoch NUM_EPOCH]
                    [--learning_rate LEARNING_RATE] [--decay_rate DECAY_RATE]
                    [--log_dir LOG_DIR]

## Output
### 1. Masking
__python train.py --kbit 10 --mask 2__

    Masking: 2
    Train: (256, 10) (256, 2)
    Test: (1024, 10) (1024, 2)

    Epoch 37/300
    Epoch 00036: val_acc did not improve
    0s - loss: 0.1172 - acc: 1.0000 - val_loss: 0.1448 - val_acc: 0.9961
    Epoch 38/300
    Epoch 00037: val_acc improved from 0.99609 to 0.99707, saving model to logs/kbit_10/weights.hdf5
    0s - loss: 0.1121 - acc: 1.0000 - val_loss: 0.1390 - val_acc: 0.9971
    Epoch 39/300
    Epoch 00038: val_acc improved from 0.99707 to 0.99902, saving model to logs/kbit_10/weights.hdf5
    0s - loss: 0.1073 - acc: 1.0000 - val_loss: 0.1336 - val_acc: 0.9990
    Epoch 40/300
    Epoch 00039: val_acc improved from 0.99902 to 1.00000, saving model to logs/kbit_10/weights.hdf5
    0s - loss: 0.1027 - acc: 1.0000 - val_loss: 0.1285 - val_acc: 1.0000

__python train.py --kbit 13 --mask 5__

    Masking: 5
    Train: (256, 13) (256, 2)
    Test: (8192, 13) (8192, 2)
    
    Epoch 79/300
    Epoch 00078: val_acc did not improve
    0s - loss: 0.0429 - acc: 1.0000 - val_loss: 0.0717 - val_acc: 0.9998
    Epoch 80/300
    Epoch 00079: val_acc did not improve
    0s - loss: 0.0419 - acc: 1.0000 - val_loss: 0.0704 - val_acc: 0.9998
    Epoch 81/300
    Epoch 00080: val_acc improved from 0.99976 to 1.00000, saving model to logs/kbit_13/weights.hdf5
    0s - loss: 0.0410 - acc: 1.0000 - val_loss: 0.0690 - val_acc: 1.0000

__python train.py --kbit 16 --mask 10__

    Masking: 10
    Train: (64, 16) (64, 2)
    Test: (65536, 16) (65536, 2)

    Epoch 00296: val_acc improved from 0.92453 to 0.92477, saving model to logs/kbit_16/weights.hdf5
    3s - loss: 0.0661 - acc: 1.0000 - val_loss: 0.1984 - val_acc: 0.9248
    Epoch 298/300
    Epoch 00297: val_acc improved from 0.92477 to 0.92513, saving model to logs/kbit_16/weights.hdf5
    3s - loss: 0.0656 - acc: 1.0000 - val_loss: 0.1977 - val_acc: 0.9251
    Epoch 299/300
    Epoch 00298: val_acc improved from 0.92513 to 0.92538, saving model to logs/kbit_16/weights.hdf5
    3s - loss: 0.0651 - acc: 1.0000 - val_loss: 0.1970 - val_acc: 0.9254
    Epoch 300/300
    Epoch 00299: val_acc improved from 0.92538 to 0.92567, saving model to logs/kbit_16/weights.hdf5
    3s - loss: 0.0646 - acc: 1.0000 - val_loss: 0.1963 - val_acc: 0.9257
    
### 2. Train/Test split (no masking)

__python train.py --kbit 13 --split_ratio 0.5__

    No masking!
    Train: (4096, 13) (4096, 2)
    Test: (4096, 13) (4096, 2)
    Total: 8192
    Ratio: 0.5

    Epoch 00000: val_acc improved from -inf to 0.90259, saving model to logs/kbit_13/weights.hdf5
    1s - loss: 0.5368 - acc: 0.7681 - val_loss: 0.4199 - val_acc: 0.9026
    Epoch 2/300
    Epoch 00001: val_acc improved from 0.90259 to 1.00000, saving model to logs/kbit_13/weights.hdf5
    0s - loss: 0.3105 - acc: 0.9744 - val_loss: 0.2066 - val_acc: 1.0000

__python train.py --kbit 13 --split_ratio 0.1__

    No masking!
    Train: (819, 13) (819, 2)
    Test: (7373, 13) (7373, 2)
    Total: 8192
    Ratio: 0.1
    
    Epoch 12/300
    Epoch 00011: val_acc improved from 0.99702 to 0.99959, saving model to logs/kbit_13/weights.hdf5
    0s - loss: 0.2676 - acc: 0.9976 - val_loss: 0.2456 - val_acc: 0.9996
    Epoch 13/300
    Epoch 00012: val_acc improved from 0.99959 to 0.99986, saving model to logs/kbit_13/weights.hdf5
    0s - loss: 0.2275 - acc: 1.0000 - val_loss: 0.2090 - val_acc: 0.9999
    Epoch 14/300
    Epoch 00013: val_acc improved from 0.99986 to 1.00000, saving model to logs/kbit_13/weights.hdf5
    0s - loss: 0.1933 - acc: 1.0000 - val_loss: 0.1781 - val_acc: 1.0000    
    
__python train.py --kbit 10 --split_ratio 0.01__
    
    No masking!
    Train: (81, 13) (81, 2)
    Test: (8111, 13) (8111, 2)
    Total: 8192
    Ratio: 0.01
    
    Epoch 128/300
    Epoch 00127: val_acc did not improve
    0s - loss: 0.1121 - acc: 1.0000 - val_loss: 0.1329 - val_acc: 0.9999
    Epoch 129/300
    Epoch 00128: val_acc improved from 0.99988 to 1.00000, saving model to logs/kbit_13/weights.hdf5
    0s - loss: 0.1105 - acc: 1.0000 - val_loss: 0.1312 - val_acc: 1.0000

### 3. Standard split (no masking)

__python train.py --kbit 10__

    No masking!
    Train: (686, 10) (686, 2)
    Test: (338, 10) (338, 2)
    Total: 1024
    Ratio: 0.67

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

    No masking!
    Train: (1372, 11) (1372, 2)
    Test: (676, 11) (676, 2)
    Total: 2048
    Ratio: 0.67

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
    
    No masking!
    Train: (2744, 12) (2744, 2)
    Test: (1352, 12) (1352, 2)
    Total: 4096
    Ratio: 0.67

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

    No masking!
    Train: (5488, 13) (5488, 2)
    Test: (2704, 13) (2704, 2)
    Total: 8192
    Ratio: 0.67

    Epoch 00000: val_acc improved from -inf to 0.90126, saving model to logs/kbit_13/weights.hdf5
    1s - loss: 0.5924 - acc: 0.7077 - val_loss: 0.4413 - val_acc: 0.9013
    Epoch 2/30
    Epoch 00001: val_acc improved from 0.90126 to 0.99963, saving model to logs/kbit_13/weights.hdf5
    0s - loss: 0.2916 - acc: 0.9750 - val_loss: 0.1678 - val_acc: 0.9996
    Epoch 3/30
    Epoch 00002: val_acc improved from 0.99963 to 1.00000, saving model to logs/kbit_13/weights.hdf5
    0s - loss: 0.1089 - acc: 1.0000 - val_loss: 0.0691 - val_acc: 1.0000
