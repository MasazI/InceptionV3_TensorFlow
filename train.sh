#########################################################################
# File Name: train.sh
# Author: yeephycho
# mail: huyixuanhyx@gmail.com
# Created Time: Tuesday, December 06, 2016 PM03:48:45 HKT
#########################################################################
#!/bin/bash


# Download 101 object categories data set
cd ./data

if [ ! -f "101_ObjectCategories.tar.gz" ]; then
    echo "Download 101_ObjectCategories dataset"
    wget https://s3-us-west-2.amazonaws.com/deep-learning.datasection.co.jp/datasets/101_ObjectCategories.tar.gz 
fi

if [ ! -d "./101_ObjectCategories/" ]; then
    PROJECT_DIR=`pwd`
    echo "Unzip 101_ObjectCategories dataset to folder "$PROJECT_DIR"/data/101_ObjectCategories/"
    tar zvxf 101_ObjectCategories.tar.gz
fi

# Generate .txt file
python create_examples_list.py
python relation_tag_to_id.py

# Start training
cd ../
#python trainer.py
