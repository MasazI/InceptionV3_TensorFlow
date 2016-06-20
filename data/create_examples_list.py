# encoding: utf-8
import glob
import os

def create_examples(directory):
    print directory
    directories = glob.glob(directory)
    for category_path in directories:
        category_name = os.path.basename(category_path) 
        print("category: %s" % (category_name))
       
        images = glob.glob("%s/%s" % (category_path, "*"))
        for image in images:
            with open('101Caltech_examples.txt', 'a') as f:
                f.write(image)
                f.write(',')
                f.write(category_name)
                f.write("\n")

if __name__ == '__main__':
    current_directory = os.getcwd()
    path = os.path.join(current_directory + "/", "101_ObjectCategories/*")
    create_examples(path) 
