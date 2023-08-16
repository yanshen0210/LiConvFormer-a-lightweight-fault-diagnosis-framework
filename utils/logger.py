#!/usr/bin/python
# -*- coding:utf-8 -*-

import logging


def setlogger(path):
    logger = logging.getLogger()  # create a logger object
    logger.setLevel(logging.INFO)  # setting the logger level
    logFormatter = logging.Formatter("%(message)s")  # output format

    fileHandler = logging.FileHandler(path)  # define a handler
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)  # add handler to logger

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)  # add handler to logger

