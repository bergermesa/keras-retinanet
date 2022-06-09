#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 8 13:39:00 2022

@author: bergermesa
"""

import os
import sys
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    import keras_retinanet  # noqa: F401
    __package__ = "keras_retinanet"

from .utils.image import preprocess_image


class Args():
    def __init__(self):
        self.batch_size = 1
        self.config = None
        self.image_min_side = 2550
        self.image_max_side = 2550
        self.no_resize = True
        self.preprocess_image = preprocess_image
        self.group_method = None


#testchange
def create_generators(args):
    
    common_args = {
            'batch_size'       : args.batch_size,
            'config'           : args.config,
            'image_min_side'   : args.image_min_side,
            'image_max_side'   : args.image_max_side,
            'no_resize'        : args.no_resize,
            'preprocess_image' : args.preprocess_image,
            'group_method'     : args.group_method
        }

    train_generator = CSVGenerator(
                args.annotations,
                args.classes,
                shuffle_groups=False,
                **common_args
            )
    return(train_generator)


args = Args()
train_generator = create_generators(args)