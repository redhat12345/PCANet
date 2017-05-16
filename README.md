# PCANet
Implementation of https://arxiv.org/pdf/1404.3606.pdf in Tensorflow

### To Replicate
download mnist, cifar10, etc... and make sure the name matches the names in the code. Run the appropriate convert scripts to converte the dataset into TFRecords. For example:

```
# with mnist folder containing the 4 mnist files
./mnist_to_record.py
```

The run PCA net. Again, make sure you set the line that picks the dataset. It looks like `load('mnist')`.

    ./pcanet.py --temp
