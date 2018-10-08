#!/bin/bash

# abort entire script on error
set -e

# train base model on mnist
python3 tools/train.py mnist train lenet lenet_mnist \
       --iterations 10000 \
       --batch_size 128 \
       --display 10 \
       --lr 0.001 \
       --snapshot 5000 \
       --solver adam

# run adda mnist->usps
python3 tools/train_adda.py mnist:train usps:train lenet adda_lenet_mnist_usps \
       --iterations 10000 \
       --batch_size 128 \
       --display 10 \
       --lr 0.0002 \
       --snapshot 5000 \
       --weights snapshot/lenet_mnist \
       --adversary_relu \
       --solver adam

# evaluate trained models
echo 'Source only baseline:'
python3 tools/eval_classification.py usps train lenet snapshot/lenet_mnist

echo 'ADDA':
python3 tools/eval_classification.py usps train lenet snapshot/adda_lenet_mnist_usps
