#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F


softmax_helper = lambda x: F.softmax(x, 1)
