# InceptionV3_TensorFlow #
InceptionV3_TensorFlow is an implementation of inception v3 using tensorflow and slim according to our guidline.


## Dependencies ##
- TensorFlow (>= 0.8)


## Features ##
- train
- save checkpoint
- real time dataaugumentation


## Setup ##
1. download data in data/readme.md
2. execute "data/create_examples_list.py"
3. execute "data/relation_tag_to_id.py"
4. you can see train_csv.txt and test_csv.txt

## Start to train##
```
python trainer.py
```


## How to use your own data sets ##
- create train_csv.txt and test_csv.txt in data directory.

### datalist format ###

```
<image path>,<label number>  
...
```
- change num_classes in settings.py

## Fine tune ##
- change fine_tune in settings.py

