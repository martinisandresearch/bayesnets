#  -*- coding: utf-8 -*-
__author__ = "Varun Nayyar <nayyarv@gmail.com>"

"""
Yaml is a little confusing - this tells you how it's being read
"""

import yaml
from pprint import pprint

with open("workflows/main.yml") as f:
    dt = yaml.safe_load(f)
pprint(dt)
