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
    #DATA_NAME = "dev-data_mini"
    DATA_NAME = "dev-data_mini-small-img"
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
        print("-----\ninspect an item of the csv generator\n-----")
        print(f"batch_size: {args.batch_size}")
        print(
            f"image_size_user_arg: {args.image_max_side} x "
            + f"{args.image_max_side}")
        
        input = item[0]
        print("\n\nGENERATOR-OUTPUT (item):\n-----\nitem[0] - INPUT:")
        print(type(input))
        print("shape: ", input.shape)
        print("-> image")

        target = item[1]
        print("\n\nitem[1] - TARGET:")
        print(type(target))
        print("len: ", len(target))
        print("-> combined reg&cls targets")
        
        reg_targets = target[0]
        print("\n\nTARGET[0] - reg-targets")
        print(type(reg_targets))
        print(reg_targets.shape)
        print(
            "regression_batch: batch that contains bounding-box regression"
            + " targets for an image & anchor states (np.array of shape "
            + "(batch_size, N, 4 + 1), where N is the number of anchors for an"
            + " image, the first 4 columns define regression targets for "
            + "(x1, y1, x2, y2) and the last column defines anchor states (-1"
            + " for ignore, 0 for bg, 1 for fg)."
        )

        cls_targets = target[1]
        print("\n\nTARGET[1] - cls-targets")
        print(type(cls_targets))
        print(cls_targets.shape)
        print(
            "labels_batch: batch that contains labels & anchor states "
            + "(np.array of shape (batch_size, N, num_classes + 1), where N is"
            + " the number of anchors for an image and the last column defines"
            + " the anchor state (-1 for ignore, 0 for bg, 1 for fg).")
        

        print(f"\nWHY is N = {cls_targets.shape[1]}?")

        anchores_N = 9
        pyramid_levels = [3, 4, 5, 6, 7]
        resolution_degradations = [2**p for p in pyramid_levels]
        pyramid_featur_map_sizes = [(2560/i) for i in resolution_degradations]
        pixels_per_pyramid_map = [s**2 for s in pyramid_featur_map_sizes]
        total_anchors = sum(pixels_per_pyramid_map * anchores_N)

        print("=>")
        print("(default) pyramid-levels are \n\t[3, 4, 5, 6, 7]")
        print("resolution-degradations are 2**pyramid-levels")
        print("thus, resolutions-degradations are:")
        print(f"\t{resolution_degradations}")
        print("image input size is:\n\t2560")
        print(
            "pyramid-feature-maps are of size image-size / "
            + "resolution-degradation"
        )
        print("thus, pyramid-feature-map-sizes are:")
        print(f"\t{pyramid_featur_map_sizes}")
        print("for every map and every pixel, all anchors are applied")
        print("pixels-per-pyramid-map is pyramid-feature-map-size**2:")
        print(f"\t{pixels_per_pyramid_map}")
        print(f"(default) anchor number is \n\t{anchores_N}")
        print("total anchors are sum(pixels-per-pyramid-map * anchors):")
        print(f"\t{total_anchors}")
        
        break

if __name__ == '__main__':
    u_debug.initialise_logging("INFO")
    main()