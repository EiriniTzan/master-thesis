"""
ae_drift_simul.py

Purpose
-------
A simulation of concept drift detection using:
1) learned representations (last hidden layer embeddings) from a small DNN
2) autoencoder reconstruction error as a drift signal (thresholding with 3-σ)

This is not the full streaming pipeline of the paper.
It's a first step to validate the core mechanism end-to-end.

"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


