"""
This file contains convenience methods and hardcoded paths depending on usage
"""
__author__ = "Varun Nayyar <nayyarv@gmail.com>"

import os

if "CI" in os.environ:
    FULL = True
else:
    print("FULL mode off")
    FULL = False
