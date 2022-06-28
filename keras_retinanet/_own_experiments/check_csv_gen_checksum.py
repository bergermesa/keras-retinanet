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
import tools.utils_tensorflow as u_tf

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
    #DATA_NAME = "dev-data_full"
    #DATA_NAME = "dev-data_mini"
    #DATA_NAME = "dev-data_mini-small-img"
    DATA_NAME = "dev-data_mini-small-imgs"
    
    BATCH_SIZE = 1

    # create bare minimum args object as needed for the fizyr implementation
    class Args():
        def __init__(self):
            self.batch_size = BATCH_SIZE
            self.config = None
            self.image_min_side = 256
            self.image_max_side = 290
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

    # process checksums
    u_tf.process_retinanet_target_checksums(
        train_generator,
        cls_target_first=False
    )


if __name__ == '__main__':
    u_debug.initialise_logging("INFO")
    main()