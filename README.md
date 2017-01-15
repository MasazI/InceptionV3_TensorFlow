# InceptionV3_TensorFlow #
InceptionV3_TensorFlow is an implementation of inception v3 using tensorflow and slim according to our guidline.


## Dependencies ##
- TensorFlow (>= 0.12)


## Features ##
- train
- save checkpoint
- real time data augumentation

## Quick start ##
If you want a quick start to run training of Inception_v3, you can simply do:
``` bash
./train.sh
```
The above script has passed test under Ubuntu15.10, CentOS and macOS.

If you want to go through the train process step by step, please take the following content as example.

### Setup ###
1. download data in data/readme.md
2. execute "data/create_examples_list.py"
3. execute "data/relation_tag_to_id.py"
4. you can see train_csv.txt and test_csv.txt

### Start to train##
```
python trainer.py
```
Pass test under Ubuntu15.10 and CentOS

### How to use your own data sets ###
- create train_csv.txt and test_csv.txt in data directory.

### datalist format ###

```
<image path>,<label number>  
...
```
- change num_classes in settings.py

### Fine tune ###
- change fine_tune in settings.py

---

Copyright (c) 2016 Masahiro Imai, Yixuan Hu (yeephycho)
Released under the MIT license
