# ML and MAP Layers for Probabilistic
In the current state of the art, SoftMax output classification networks are usually implemented for perception systems and applied in autonomous navigation, including mobile robotics and autonomous vehicles. However, the nature of
SoftMax classification often produces overconfident predictions rather than proper probabilistic interpretation, which can thus hinder the ability of the classification network. In this work, we
propose a probabilistic approach to Convolutional Neural Networks which is calculated via the distributions in the network’s Logit layer. We demonstrate that Maximum Likelihood (ML)
and Maximum a-Posteriori (MAP) interpretations are stronger for classification than the SoftMax layer by performing entity recognition within the KITTI and Lyft Level-5 (LL5) datasets.
We explore the modalities of computer vision via an RGB camera and LiDAR (RV: range-view) modalities, where our approach shows promising performance compared to the usual SoftMax
layer with the benefit of enabling interpretable probabilistic predictions. Another benefit to our proposed system is that the layers can be implemented as a replacement to SoftMax, that is,
they can be used as classification output to pre trained networks.
