#!/bin/bash

# abort entire script on error
set -e

# train base model on usps
python3 tools/train.py usps train lenet lenet_usps \
       --iterations 10000 \
       --batch_size 128 \
       --display 10 \
       --lr 0.001 \
       --snapshot 5000 \
       --solver adam

# run adda usps->mnist
python3 tools/train_adda.py usps:train mnist:train lenet adda_lenet_usps_mnist \
       --iterations 10000 \
       --batch_size 128 \
       --display 10 \
       --lr 0.0002 \
       --snapshot 5000 \
       --weights snapshot/lenet_usps \
       --adversary_relu \
       --solver adam

# evaluate trained models
echo 'Source only baseline:'
python3 tools/eval_classification.py mnist train lenet snapshot/lenet_usps

echo 'ADDA':
python3 tools/eval_classification.py mnist train lenet snapshot/adda_lenet_usps_mnist
