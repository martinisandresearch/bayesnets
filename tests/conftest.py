#  -*- coding: utf-8 -*-
__author__ = "Varun Nayyar <nayyarv@gmail.com>"

import os
import sys

# convenience for testing
swarmpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if swarmpath not in sys.path:
    sys.path.insert(0, swarmpath)