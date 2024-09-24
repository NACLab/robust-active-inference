"""
Module: aif
Author: Viet Nguyen
Date: 2023-01-02

Description: This module implement an active inference agent that make use of the
  Contrastive Recurrent State Prior Preference model. This is the main method/agent of our paper
"""

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent))

from .agent import Agent

