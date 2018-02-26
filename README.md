# Reference Library
My notes about reference informations.

## Basic Knowledge
### Books
- [Stanford University C231n](http://cs231n.github.io/)
- [Deep Learning (MIT)](http://www.deeplearningbook.org/)
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)

### Weight initialization
- [Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) by Xavier Glorot et al.
- [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf) by K. He et al.

### Optimization
- [Efficient BackProp](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf) by Y. LeCun el al.
- [Adaptive Subgradient Methods for Online Learning and Stochastic Optimization (Adagrad)](http://jmlr.org/papers/v12/duchi11a.html) by J. Duchi et al.
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) by D. P. Kingma et al.
- [RMSprop](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
- Nesterov Momentum
  - [Advances in optimizing Recurrent Networks](https://arxiv.org/pdf/1212.0901v2.pdf) by Y. Bengio et al., Section 3.5.
  - [Ilya Sutskever’s thesis](http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf) (pdf), section 7.2

### Batch Normalization
- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
- [Why does batch normalization help?](https://www.quora.com/Why-does-batch-normalization-help)

### Regularization
- Dropout
  - [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
- DropConnect
  - [Regularization of Neural Networks using DropConnect](https://cs.nyu.edu/~wanli/dropc/)

### Hyper-Parameter Optimization
- [Random Search for Hyper-Parameter Optimization](http://www.jmlr.org/papers/v13/bergstra12a.html) by J.Bergstra et al.
- [Systematic evaluation of CNN advances on the ImageNet](https://arxiv.org/abs/1606.02228) by D. Mishkin et al.
  - In this article, they mention about the mini-batch size dependency
- [Practical recommendations for gradient-based training of deep architectures](https://arxiv.org/abs/1206.5533) by Yoshua Bengio
- [Neural Networks and Deep Learning book - Chapter 3: How to choose a neural network's hyper-parameters?](http://neuralnetworksanddeeplearning.com/chap3.html#how_to_choose_a_neural_network's_hyper-parameters)  by Michael Nielsen

## CNN
### Object Recognition
- LeNet
  - [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) by Y. LeCun et al., 1998.
- AlexNet
  - [ImageNet Classification with Deep Convolutional Neural Networks](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) by Alex Krizhevsky et al., 2012.
- VGG
  - [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) by K. Simonyan et al., 2014.
- GoogLeNet
  - [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842) by C. Szegedy et al., 2014.
- Network In Network
  - [Network In Network](https://arxiv.org/pdf/1312.4400.pdf) by M. Lin et al., 2014
    - This paper mentions about Global Average Pooling (GAP).
- ResNet
  - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) by K. He et al., 2015.
- Inception
  - [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567) by C. Szegedy et al., 2015.
- Xception
  - [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357) by F. Chollet (Google), 2016.
- DenseNet
  - [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) by G. Huang et al., 2016.
- MobileNet
  - [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861) by A. G. Howard et al., 2017.
  
### Transfer Learning
- [How transferable are features in deep neural networks?](https://arxiv.org/pdf/1411.1792.pdf)

### Visualize CNN
- Class activation maps
  - Learning Deep Features for Discriminative Localization [[Paper]](https://arxiv.org/abs/1512.04150) [[website]](http://cnnlocalization.csail.mit.edu/) by B. Zhou et al., 2015.
  
### Object Detection
- R-CNN
  - [Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/abs/1311.2524) by R. Girshick et al., 2013.
- Fast R-CNN
  - [Fast R-CNN](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf) by R. Girshick et al.
- Faster R-CNN
  - [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497) by S. Ren et al., 2015.

### Segmentation
- FCN
  - [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038) by J. Long et al., 2014.
- SegNet
  - [SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation](https://arxiv.org/abs/1511.00561) by V. Badrinarayanan et al., 2015.
- Dilated Convolution
  - [Multi-Scale Context Aggregation by Dilated Convolutions](https://arxiv.org/abs/1511.07122) by F. Yu et al., 2015.
- U-Net
  - [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) by O. Ronneberger et al., 2015.
- Deep lab. (v1 & v2)
  - [Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs](https://arxiv.org/abs/1412.7062) by L. C. Chen et al., 2014.
  - [DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs](https://arxiv.org/abs/1606.00915) by L. C. Chen et al., 2016.
- RefineNet
  - [RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation](https://arxiv.org/abs/1611.06612) by G. Lin et al., 2016.
- PSPNet
  - [Pyramid Scene Parsing Network](https://arxiv.org/abs/1612.01105) by H. Zhao et al., 2016.
- Large Kernel Matters
  - [Large Kernel Matters -- Improve Semantic Segmentation by Global Convolutional Network](https://arxiv.org/abs/1703.02719) by C. Peng et al., 2017.
- DeepLab v3
  - [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587) by L. C. Chen et al., 2017.

### RNN & LSTM
- RNN
  - [TDNN (Time delay neural network) wiki](https://en.wikipedia.org/wiki/Recurrent_neural_network#Elman_networks_and_Jordan_networks)
  - [Elman Network](http://onlinelibrary.wiley.com/doi/10.1207/s15516709cog1402_1/abstract)
  - [RNN wiki](https://en.wikipedia.org/wiki/Recurrent_neural_network#Elman_networks_and_Jordan_networks)
- LSTM
  - [LONG SHORT-TERM MEMORY](http://www.bioinf.jku.at/publications/older/2604.pdf) by Sepp Hochreiter and Jürgen Schmidhuber.
- RNN Hyperparamters (model structures)
  - [How to Generate a Good Word Embedding?](https://arxiv.org/abs/1507.05523) by S. Lai et al.
  - [Visualizing and Understanding Recurrent Networks](https://arxiv.org/abs/1506.02078) by A. Karpathy et al.

### Style Transfer
(editing)
### Image Generation
(editing)

## Websites & blogs
- CNN
  - [Large Scale Image Recognition using CNN's](https://arxiv.org/pdf/1409.1556.pdf)
  - [CNN Tricks](http://lamda.nju.edu.cn/weixs/project/CNNTricks/CNNTricks.html)
  - [A guide to receptive field arithmetic for Convolutional Neural Networks](https://medium.com/@nikasa1889/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807)
  
- Optimizer
  - [An overview of gradient descent optimization algorithms](http://ruder.io/optimizing-gradient-descent/index.html)
  
- Segmentation
  - [A 2017 Guide to Semantic Segmentation with Deep Learning](http://blog.qure.ai/notes/semantic-segmentation-deep-learning-review)

- RNN & LSTM
  - [A Beginner’s Guide to Recurrent Networks and LSTMs](https://deeplearning4j.org/lstm.html)
  - [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
  - [Exploring LSTMs](http://blog.echen.me/2017/05/30/exploring-lstms/)
  - [Learning Long-Term Dependencies with RNN](http://www.cs.toronto.edu/~guerzhoy/321/lec/W09/rnn_gated.pdf)
  
- Keras
  - [How to Grid Search Hyperparameters for Deep Learning Models in Python With Keras](https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/)
  - [How to Check-Point Deep Learning Models in Keras](https://machinelearningmastery.com/check-point-deep-learning-models-keras/)
  - [Class activation maps in Keras for visualizing where deep learning networks pay attention](https://jacobgil.github.io/deeplearning/class-activation-maps)
  - [data augmentation](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)
  - [Building an Image Classifier](https://towardsdatascience.com/learning-about-data-science-building-an-image-classifier-part-2-a7bcc6d5e825)
    - This blog mentions about transfer learning with Keras.
    
- Tensorflow
  - [Google Developers Blog](https://developers.googleblog.com/search/label/TensorFlow)
  - [How to improve my test accuracy using CNN in Tensorflow](https://datascience.stackexchange.com/questions/20104/how-to-improve-my-test-accuracy-using-cnn-in-tensorflow)
  
## My Memo
- [Stanford Vision Lab.](http://vision.stanford.edu/publications.html)
- ISIC Challenge 2017
  - [SKIN LESION ANALYSIS TOWARD MELANOMA DETECTION: A CHALLENGE AT THE 2017 INTERNATIONAL SYMPOSIUM ON BIOMEDICAL IMAGING (ISBI), HOSTED BY THE INTERNATIONAL SKIN IMAGING COLLABORATION (ISIC)](https://arxiv.org/pdf/1710.05006.pdf)
  - [Image Classification of Melanoma, Nevus and Seborrheic Keratosis by Deep Neural Network Ensemble](https://arxiv.org/ftp/arxiv/papers/1703/1703.03108.pdf)
  - [Incorporating the Knowledge of Dermatologists to Convolutional Neural Networks for the Diagnosis of Skin Lesions](https://arxiv.org/pdf/1703.01976.pdf)
  - [RECOD Titans at ISIC Challenge 2017](https://arxiv.org/abs/1703.04819)
## Datasets or Competition
- [List of datasets for machine learning research (wikipedia)](https://en.wikipedia.org/wiki/List_of_datasets_for_machine_learning_research)
- [Kaggle](https://www.kaggle.com/)
- [Deep Analytics](https://deepanalytics.jp/?tc=menu)
