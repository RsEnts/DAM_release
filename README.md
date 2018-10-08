# DAM_release

Implementation for **Domain transfer through deep Activation Matching**.

> Haoshuo Huang, Qixing Huang, Philipp Krähenbühl,     
> **Domain transfer through deep activation matching**      
> ECCV 2018 (http://www.philkr.net/papers/2018-09-01-eccv/2018-09-01-eccv.pdf)

Contact: [haosoul122@gmail.com](mailto:haosoul122@gmail.com). 

We have release the implementation in digital image classification task, the segmentation task need more engineering work and will release soon.

## Abstract 

We introduce a layer-wise unsupervised domain adaptation approach for semantic segmentation. Instead of merely matching the output distributions of the source and target domains, our approach aligns the distributions of activations of intermediate layers. This scheme exhibits two key advantages. First, matching across intermediate layers introduces more constraints for training the network in the target domain, making the optimization problem better conditioned. Second, the matched activations at each layer provide similar inputs to the next layer for both training and adaptation, and thus alleviate covariate shift. We use a Generative Adversarial Network (or GAN) to align activation distributions. Experimental results show that our approach achieves stateof-the-art results on a variety of popular domain adaptation tasks, including (1) from GTA to Cityscapes for semantic segmentation, (2) from SYNTHIA to Cityscapes for semantic segmentation, and (3) adaptations on USPS and MNIST for image classification.

## Citation
    @InProceedings{huang2018ddam,
    author = {Huang, Haoshuo and Huang, Qixing and Krähenbühl, Philipp},
    title = {Domain transfer through deep Activation Matching},
    journal={European Conference on Computer Vision (ECCV)},
    year={2018}
    }