# DAM for digits classification

## Introduction

In the digits classification settings, we borrow the implementation of ADDA (https://github.com/erictzeng/adda). Then we apply our DAM algorithm based on their implementation.
 The result shows that our ADDA can get 95% from MNIST to USPS, 98% from USPS to MNIST and 74 from SVHN to MNIST.

## Getting started

This part is the same as ADDA.

Simply run:

    pip install -r requirements.txt
    mkdir data snapshot
    export PYTHONPATH="$PWD:$PYTHONPATH"
    scripts/mnist-usps.sh

to get the final prediction.