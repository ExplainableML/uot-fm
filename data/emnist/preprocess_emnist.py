import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def flip_rotate(image):
    W = 28
    H = 28
    image = image.reshape(W, H)
    image = np.rot90(np.rot90(np.rot90(image)))
    image = np.fliplr(image)
    return image.reshape(1, W, H)


if __name__ == "__main__":
    train_bymerge_df = pd.read_csv("archive/emnist-byclass-train.csv", header=None)
    test_bymerge_df = pd.read_csv("archive/emnist-byclass-test.csv", header=None)
    
    label_map = pd.read_csv("archive/emnist-byclass-mapping.txt", delimiter = ' ', index_col=0, header=None).squeeze("columns")
    label_dictionary = {}
    for index, label in enumerate(label_map):
        label_dictionary[index] = chr(label)

    train_bymerge_df = train_bymerge_df.rename(columns= {0: 'label'})
    test_bymerge_df = test_bymerge_df.rename(columns= {0: 'label'})

    # filter for [0, 1, 8] & [O, I, B]
    acceptable_range = [0, 1, 8, 11, 18, 24]

    train_data = train_bymerge_df[train_bymerge_df.label.isin(acceptable_range)]
    test_data = test_bymerge_df[test_bymerge_df.label.isin(acceptable_range)]
    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)

    y = np.array(train_data['label'])
    x = np.array(train_data.drop(columns=['label']))
    y_test = np.array(test_data['label'])
    x_test = np.array(test_data.drop(columns=['label']))

    x_test = np.array(list(map(flip_rotate, x_test)))
    x = np.array(list(map(flip_rotate, x)))

    np.save("x_train.npy", x)
    np.save("y_train.npy", y)
    np.save("x_test.npy", x_test)
    np.save("y_test.npy", y_test)