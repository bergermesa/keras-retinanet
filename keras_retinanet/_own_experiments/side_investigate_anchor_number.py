#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 8 13:39:00 2022

@author: bergermesa
"""

anchores_N = 9

sizes   = [32, 64, 128, 256, 512]
strides = [8, 16, 32, 64, 128]

pyramid_levels = [3, 4, 5, 6, 7]
resolutions = [2**p for p in pyramid_levels] # == strides


n = [(2560/i)**2 for i in resolutions]
print(sum(n)*9)

"""
--------
From netron inspection:
--------
P3 has resolution: image / (2*2*2 [strides])
                      -> image/8
P4 has resolution: image / (2*2*2*2 [strides])
                      -> image/16
P5 has resolution: image / (2*2*2*2*2 [strides])
                      -> image/32
P6 has resolution: image / (2*2*2*2*2*2 [strides])
                      -> image/64
P7 has resolution: image / (2*2*2*2*2*2*2 [strides])
                      -> image/128
"""