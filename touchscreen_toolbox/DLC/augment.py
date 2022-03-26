# file to create augmented dataset

# NOT TESTED YET

import os
import sys
import cv2
import shutil
import numpy as np
import pandas as pd
from time import sleep


def xflip(row, dim=(640, 480)):
    """
    Flip labels along y-axis (horizontally)

    Args:
    ------
    row : ndarray
        row of labels (x,y coordinates)

    dim : tuple, default = (640,480)
        resolution of the video
    """
    new_row = np.copy(row)
    new_row[::2] = dim[0] - new_row[::2]
    return new_row


def yflip(row, dim=(640, 480)):
    """Flip labels along x-axis (vertically)"""
    new_row = np.copy(row)
    new_row[1::2] = dim[1] - new_row[1::2]
    return new_row


def random_exposure(a=0.4, b=0.4, l_bound=0.5):
    """
    Randomly create a lut table, that adjust exposure of the image
    (y/255) = (x/255)^gamma, gamma > 0
    gamma is sampled from a beta distribution

    Args
    ------
    a : float
        parameter a for the distribution

    b : float
        parameter b

    l_bound : float
        lower bound to be added to the random number from beta

    Returns
    ------
    lut_table : ndarray
        Lut table in (1 x 256) numpy array

    """

    gamma = np.random.beta(a, b) + l_bound
    lut_table = np.power(np.arange(256) / 255, gamma) * 255
    lut_table = np.round(lut_table).astype(np.uint8)
    return lut_table


def lut(frame, lut_table):
    """
    Map LUT table to the image

    Args:
    ------
    frame : ndarray
        image to be mapped

    lut_table : ndarray (1 x 256)
        lut table to use

    """
    return cv2.LUT(frame, lut_table)


def augment(ls, img, row, flipcode, prefix):
    """
    Args
    ------
    ls : list
        list to save elements

    img : ndarray
        image

    row : ndarray
        labels for the image

    flipcode : int
        flipcode, see documentation for 'cv2.flip'

    prefix : str
        path to save flipped image

    Returns
    ------
    None,
    - augmented image is wrote into the disk
    - corresponding labels are saved into the [ls]

    """

    # augment
    new_img = cv2.flip(img, flipcode)  # flip
    lut_table = random_exposure()
    new_img = lut(new_img, lut_table)  # adjust exposure

    # metadata
    if flipcode == 1:     # horizontal
        new_row = xflip(row)
        new_name = prefix + "_h.png"

    elif flipcode == 0:   # vertical
        new_row = yflip(row)
        new_name = prefix + "_v.png"

    elif flipcode == -1:  # both
        new_row = xflip(yflip(row))
        new_name = prefix + "_d.png"

    else:
        raise ValueError("Invalid flipcode")

    # save
    cv2.imwrite(new_name, new_img)
    ls.append((new_name, new_row))


if __name__ == "__main__":

    # magic strings
    root = 'labeled-data'
    path2save = root + "\\aug"
    template_file = "labeled-data\\1\\CollectedData_Harris.csv"

    # make directory for augmented data
    if os.path.exists(path2save):
        print("Removing existing aug folder")
        shutil.rmtree(path2save)
        try:
            shutil.rmtree(path2save + "_labeled")
        except BaseException:
            pass
    sleep(20)  # wait for google drive to sync...
    os.makedirs(path2save)

    # list to store elements for labels.csv
    new_csv = []

    # iterate thru sub-directories
    for path, sub, files in os.walk(root):

        # filter out non-leaf directories
        if not sub:

            # find the labels.csv file
            csv = [f for f in files if f.endswith('csv')]

            # ignore irrelevant folders
            if csv:

                print(f"Processing {path} : ", end='')
                csv = os.path.join(path, csv[0])
                imgs = sorted([i for i in files if 'png' in i])
                lbls = pd.read_csv(
                    csv, header=None, skiprows=[
                        0, 1, 2, 3]).set_index(0)

                # augment each image
                for (idx, row), img_name in zip(lbls.iterrows(), imgs):

                    # check file name
                    img_path = idx.split("\\")
                    # assert img_path[-1] == img_name, "Unmatched image file"
                    prefix = path2save + "\\" + \
                        ("-".join(img_path[-2:]))[:-
                                                  4]  # file extension removed

                    # read image
                    img = cv2.imread(os.path.join(path, img_name))

                    # augment
                    augment(new_csv, img, row, 0, prefix)
                    augment(new_csv, img, row, 1, prefix)
                    augment(new_csv, img, row, -1, prefix)

                    print("█", end='')

                print("\n")

    # format output csv
    # copy a template for DeepLabCut to read
    headers = pd.read_csv(template_file, header=None).iloc[:4].values

    # rearrange elements and form the new csv file
    indice = np.array(list(zip(*new_csv))[0], ndmin=2).T
    values = np.array(list(zip(*new_csv))[1])

    # combine everything
    new_values = np.hstack((indice, values))
    new_values = np.vstack((headers, new_values))
    output = pd.DataFrame(new_values)

    # save
    output.to_csv(
        path2save +
        "\\" +
        "CollectedData_Harris.csv",
        header=False,
        index=False)

    # calling DeeplabCut to make csv into h5 files
    from deeplabcut import convertcsv2h5
    convertcsv2h5('config.yaml', userfeedback=False)
    print('done')
