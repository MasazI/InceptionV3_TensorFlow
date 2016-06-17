# encoding: utf-8

import numpy as np

def main(filepath):
    data = np.loadtxt(filepath, delimiter=",", dtype=np.str)
    id_index = -1
    classes = set()
    datasets = {}
    for line in data:
        image = line[0]
        label = line[1]
        if not label in classes:
            classes.add(label)
            id_index += 1
            with open('classes.txt', 'a') as f:
                f.write(label)
                f.write(',')
                f.write(str(id_index))
                f.write('\n')

        datasets[image] = id_index

    with open('train_csv.txt', 'w') as f:
        for key, value in datasets.iteritems():
            f.write(key)
            f.write(',')
            f.write(str(value))
            f.write("\n")


if __name__ == '__main__':
    main('101Caltech_examples.txt')