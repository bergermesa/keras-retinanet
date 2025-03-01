#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 8 13:39:00 2022

@author: bergermesa
"""
# python packages
import logging
import numpy as np
import os
import sys
import time
from tqdm import tqdm

# own packages
import tools.debug as u_debug

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    import keras_retinanet._own_experiments  # noqa: F401
    __package__ = "keras_retinanet._own_experiments"

# ensure the cython script gets built
import pyximport
pyximport.install(
    setup_args={"include_dirs":np.get_include()},
    reload_support=True
)

from ..preprocessing.csv_generator import CSVGenerator
from ..utils.image import preprocess_image



def main():
    # User inputs
    DATA_NAME = "dev-data_mini"
    BATCH_SIZE = 1

    # create bare minimum args object as needed for the fizyr implementation
    class Args():
        def __init__(self):
            self.batch_size = BATCH_SIZE
            self.config = None
            self.image_min_side = 2560
            self.image_max_side = 2560
            self.no_resize = True
            self.preprocess_image = preprocess_image
            self.group_method = None
            base_path = (
                "/Users/marlinberger/Desktop/local/Coding/mesasight/"
                + "keras-retinanet/keras_retinanet/_own_experiments/"
                + f"data/{DATA_NAME}/"
                )
            self.annotations = base_path + "annotations.csv"
            self.classes = base_path + "classes.csv"

    # imitate fizyr 'create_generators' method as bare minimum
    def create_generators(args):
        # initialise the args needed for the CSVGenerator
        common_args = {
                'batch_size'       : args.batch_size,
                'config'           : args.config,
                'image_min_side'   : args.image_min_side,
                'image_max_side'   : args.image_max_side,
                'no_resize'        : args.no_resize,
                'preprocess_image' : args.preprocess_image,
                'group_method'     : args.group_method
            }
        # initialise the actual generator
        train_generator = CSVGenerator(
                    args.annotations,
                    args.classes,
                    shuffle_groups=False,
                    **common_args
                )
        return(train_generator)

    # get args and train-generator
    args = Args()
    train_generator = create_generators(args)

    for item in train_generator:
        # these numbers shall match in with own implementation, when using the
        # same:
        #       data-base
        #       batch-size
        #       pyramid-levels
        #       anchor-ratios
        #       anchor-scales
        print("\ntotal anchor number:")
        print(item[1][0].shape[1])

        print(f"\nreg-target shape:\n{item[1][0].shape}")
        print("sum over reg-labels batch & axis=1:")
        print(np.sum(np.sum(item[1][0], axis=1), axis=0))

        print(f"\ncls-target shape:\n{item[1][1].shape}")
        print("sum over cls-labels batch & axis=1:")
        print(np.sum(np.sum(item[1][1], axis=1), axis=0))

        print(
            "\nNOTE: Last number of check_sums from reg&cls MUST match, as"
            + "as these are the markers, if an anchor is fg, bg or ignored"
            + "\n\nThe cls numbers [:-1] say, how many anchors in the batch are"
            + " assigned to the specific class"
            + "\n\nThe reg number [:-1] are the checksums of bboxes"
        )

        break

if __name__ == '__main__':
    u_debug.initialise_logging("INFO")
    main()