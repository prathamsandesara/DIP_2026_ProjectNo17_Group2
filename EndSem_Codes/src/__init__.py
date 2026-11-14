"""
fabric-defect-dip
==================

A classical Digital Image Processing (DIP) pipeline for adaptive fabric
surface defect detection and localization using texture descriptors
(Local Binary Patterns, Gray-Level Co-occurrence Matrix / Haralick features,
and Gabor filtering) evaluated on the AITEX Fabric Image Database (AFID).

This package intentionally contains NO deep learning. Every stage is a
classical, explainable image-processing operation, as required for a
Digital Image Processing course project.

Sub-packages
------------
data          : dataset parsing, splitting, validation
preprocessing : grayscale conversion, filtering, normalization, enhancement
texture       : LBP, GLCM/Haralick, Gabor, local statistics, multi-scale
adaptive      : adaptive scale estimation, parameter selection, thresholding
detection     : normal texture modeling, deviation maps, fusion, segmentation
morphology    : binary morphological post-processing
localization  : connected-component defect localization, overlays
evaluation    : pixel/image/fabric/defect-type metrics
utils         : I/O, visualization, logging helpers
"""

__version__ = "0.1.0"
