#!/bin/csh

set PYTHON_MODULE              = dev_tools/oct2018/python-Anaconda-3-5.3.0
set CHAINCONSUMER_MODULE       = astro/dec2019/chainconsumer-0.30
set UNNEEDED_PYTHON_MODULE     = dev_tools/oct2017/python-Anaconda-3-5.0.0.1 # Loaded by ChainConsumer module file...

module purge

module load $PYTHON_MODULE
module load $CHAINCONSUMER_MODULE

module unload $UNNEEDED_PYTHON_MODULE
