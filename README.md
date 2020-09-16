# [DeTraC](https://arxiv.org/pdf/2003.13815.pdf)

## Usage

Before running the script, run ```pip install -r requirements.txt``` so that you can install all the necessary dependencies. Also, make sure that you are located in the ```src``` directory inside the ```DeTraC``` folder.

A key aspect to keep in mind is that this script can be used with either a TensorFlow backend, or a PyTorch backend. Regardless of the chosen framework, the results should be the same. 

To select a framework use the ```-f``` flag, followed by the framework of choice:
```bash
python detrac.py -f tf
```

OR

```bash
python detrac.py -f torch
```

This script consists of two elements:
- Training:
    ```bash
    python detrac.py -f <FRAMEWORK> --train --epochs <NUMBER_OF_EPOCHS> --num_classes <NUMBER_OF_CLASSES> --batch_size <BATCH_SIZE> --folds <NUMBER_OF_FOLDS> --k <NUMBER_OF_CLUSTER(K-Means)> --lr <FEATURE_EXTRACTOR_LR> <FEATURE_COMPOSER_LR>
    ```

- Inference
    ```bash
    python detrac.py -f <FRAMEWORK> --infer
    ```

You can also run the script with both the ```--infer``` and ```--train``` flag simultaneously:
```bash
python detrac.py -f <FRAMEWORK> --infer --train [...]
```

For details regarding the flags, you can use the ```-h / --help``` flag:
```bash
python detrac.py -h
```

## Routine

### <b>Initialization</b>

When you first run the script, if the ```data``` and ```models``` directories are not created in your ```DeTraC``` folder, they will be created by the script and it will not proceed until you populate the ```data/initial_data``` directory with your data.

The image data should be located in a folder corresponding with one of the classes you wish to train the model on.

For example:
> ```data/initial_dataset/COVID19/covid_image.jpg```<br>
> ```data/initial_dataset/PNEUMONIA/pneumonia_image.jpg```<br>
> ```data/initial_dataset/NORMAL/normal_image.jpg```<br>

### <b>Training</b>

The DeTraC model consists of training one model (we'll call it the ```feature extractor```), using said model to extract features, with which we'll compose a new dataset of images using <i>k-means clustering</i> to divide the features into k parts, and training a new model (we'll call it the ```feature composer```) on the newly composed dataset.

Both the feature extractor and the feature composer are based on a pretrained model (this particular script uses the VGG16 model).

#### <b>Feature extractor model training</b>

Before training, we preprocess every image in the initial dataset and label it according to it's class name using <i>one-hot encoding</i>

We then split those features and labels in folds, using the k-fold validation split method:

> Let K be the number of folds (an integer)<br>
> Training Set = <i>100% - (K * 10)%</i> | <i>Validation Set = (K * 10)%</i>

Afterwards, we normalize the training and validation feature sets.

Then, the feature extractors training process starts

It is worth mentioning that there are 3 possible modes of training:
- Shallow tuning | All pretrained layers have their weights frozen and the custom classification layer has its weights active
- Deep tuning | All layers have their weights active
- Fine tuning | A certain number of low-level layers have their weights frozen and the rest have their weights active

Here's a sample output:
```
# Using PyTorch backend as an example
# Training is done on 1 epoch for the purpose of demonstration

[PyTorch Backend]

Preparing the model for training
Use CUDA for GPU computation? [Y / N]: y
Loading images from directory COVID19: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 576/576 [01:11<00:00,  8.07it/s]
Loading images from directory NORMAL: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 317/317 [00:35<00:00,  8.92it/s]
Loading images from directory PNEUMONIA: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 2358/2358 [02:56<00:00, 13.37it/s]
            
            
            Choose a mode in which you wish to train:
            1) Shallow-tuning (Fast, but inaccurate)
            2) Deep-tuning (Slow and requires a lot of data, but accurate)
            3) Fine-tuning
            
> 3
Pretrained model has 37 layers.
> How many layers to train?: 25  
Freezing 12 layers and activating 25.
[Epoch 1 stats]: train_loss = 0.78 | train_acc = 72.05% | val_loss = 0.74 | val_acc = 70.54%: 100%|███████████████████████████████████████████████| 1/1 [00:58<00:00, 58.58s/it]
[[0.         0.         0.18190212]
 [0.         0.         0.11265005]
 [0.         0.         0.70544783]]
ACCURACY = 0.8036318867343799
```

#### <b>Feature extraction process</b>

After training is done, the feature extractor will use the same images from the initial dataset and it will predict ```Nx4096``` features from them (inference is done without last layer of classification), where ```N = Number of images in a folder```.

Sample output:
```
Preparing COVID19 for feature extraction: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 576/576 [00:58<00:00,  9.85it/s]
Preparing NORMAL for feature extraction: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 317/317 [00:28<00:00, 11.08it/s]
Preparing PNEUMONIA for feature extraction: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 2358/2358 [02:24<00:00, 16.32it/s]
```

#### <b>Dataset composition process</b>

After feature extraction, the features from each class are taken and are divided using 2-means clustering. Then, each image is loaded and is placed in its corresponding class folder.

Sample output:
```
Directory COVID19_1 already exists. Overwriting.
Directory COVID19_2 already exists. Overwriting.
Composing COVID19 images: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 576/576 [02:19<00:00,  4.14it/s]
Directory NORMAL_1 already exists. Overwriting.
Directory NORMAL_2 already exists. Overwriting.
Composing NORMAL images: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 317/317 [01:11<00:00,  4.44it/s]
Directory PNEUMONIA_1 already exists. Overwriting.
Directory PNEUMONIA_2 already exists. Overwriting.
Composing PNEUMONIA images: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 2358/2358 [05:05<00:00,  7.73it/s]
```

#### <b>Feature composer model training</b>

Here is where the training process is similar to the feature extractor's training process, except for the fact that this time we <b>activate all weights in order to compute their gradients</b> (enable learning).

Sample output:
```
Loading images from directory COVID19_1: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 246/246 [00:17<00:00, 13.77it/s]
Loading images from directory COVID19_2: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 330/330 [00:27<00:00, 11.84it/s]
Loading images from directory NORMAL_1: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 142/142 [00:10<00:00, 13.69it/s]
Loading images from directory NORMAL_2: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 175/175 [00:14<00:00, 12.11it/s]
Loading images from directory PNEUMONIA_1: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 1147/1147 [01:11<00:00, 16.15it/s]
Loading images from directory PNEUMONIA_2: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 1211/1211 [01:16<00:00, 15.74it/s]
[Epoch 1 stats]: train_loss = 1.50 | train_acc = 44.60% | val_loss = 1.40 | val_acc = 49.77%: 100%|██████████████████████████████████████████████| 1/1 [01:46<00:00, 106.91s/it]
[[0.         0.         0.19113573]
 [0.         0.         0.10249307]
 [0.         0.         0.70637119]]
ACCURACY = 0.8042474607571561
```

### <b>Inference</b>

When using the model for inference, you'll be asked to give the path of the image you wish to use the model upon, as such:

```
Please enter the path of the file you wish to run the model upon (e.g.: /path/to/image.png): ../data/initial_dataset/COVID19/COVID19(0).jpg
../data/initial_dataset/COVID19/COVID19(0).jpg
```

Afterwards, you'll be asked to choose one of the models you wish to use for inference, as such:

```
Here is a list of your models: 
1) DeTraC_feature_composer_2020-09-06_15-25-31.pth
2) DeTraC_feature_composer_2020-09-06_16-25-58.pth
Which model would you like to load? [Number between 1 and 2]: 2
```

Here is the results after choosing your model:
```
Prediction: COVID19
Confidence: 
{'COVID19_1': 0.12374887, 'COVID19_2': 0.16667016, 'NORMAL_1': 0.093206584, 'NORMAL_2': 0.09856959, 'PNEUMONIA_1': 0.2301002, 'PNEUMONIA_2': 0.28770462}
```


### <b>Citation</b>


If you used DeTraC and found it useful, please cite the following papers:

• Abbas A, Abdelsamea MM, Gaber MM. DeTraC: Transfer Learning of Class Decomposed Medical Images in Convolutional Neural Networks. IEEE Access 2020. (https://ieeexplore.ieee.org/document/9075155?source=authoralert)

• Abbas A, Abdelsamea MM, Gaber MM. Classification of COVID-19 in chest X-ray images usingDeTraC deep convolutional neural network. Applied Intelligence, 2020. (https://link.springer.com/article/10.1007/s10489-020-01829-7)
