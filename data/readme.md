inputs data format: csv  
each raw: <imagepath>,<label number>  

samples: Caltech101  
https://s3-us-west-2.amazonaws.com/deep-learning.datasection.co.jp/datasets/101_ObjectCategories.tar.gz  
unzip the file, you can get 101_ObjectCategories directory.  
  
use script:  
create_examples_list.py --> image path,category list  
relation_tag_to_id.py --> image path,label list to train  
