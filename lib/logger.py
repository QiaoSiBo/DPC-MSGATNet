#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging


# logging level: DEBUG < INFO < WARNING < ERROR < CRITICAL
def setlogger(path):
    logger = logging.getLogger()  # create a Logger (创建一个日志记录器)
    logger.setLevel(logging.INFO)  # set Logger level (设置日志级别，大于该级别的日志记录都能输出，此时日志级别为INFO)
    logFormatter = logging.Formatter("%(asctime)s %(message)s", "%m-%d %H:%M:%S")  # set log format

    fileHandler = logging.FileHandler(path)  # 创建一个文件处理器， 用于将日志保存至文件中
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)  # 为日志记录器添加一个文件处理器

    consoleHandler = logging.StreamHandler()  #创建一个流处理器，用于将INFO以上日志输出到终端
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)


if __name__ == '__main__':
    path = 'test.log'
    setlogger(path)
    logging.info('Epoch: ' + '-'*5)