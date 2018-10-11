# Reference Library
My notes about reference information.

## Books and Documents
- [Stanford University C231n](http://cs231n.github.io/)
- [Deep Learning Book](http://www.deeplearningbook.org/)
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)

## Basic Knowledge
### Weight Initialization
- [Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) by Xavier Glorot et al.
- [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf) by K. He et al.

### Activation
- [Maxout Networks](https://arxiv.org/abs/1302.4389) by Ian J. Goodfellow et al., 2013.

### Optimization
- [An overview of gradient descent optimization algorithms](http://ruder.io/optimizing-gradient-descent/index.html)
- [Efficient BackProp](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf) by Y. LeCun el al.
- [Adaptive Subgradient Methods for Online Learning and Stochastic Optimization (Adagrad)](http://jmlr.org/papers/v12/duchi11a.html) by J. Duchi et al.
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) by D. P. Kingma et al.
- [RMSprop](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
- Nesterov Momentum
  - [Advances in optimizing Recurrent Networks](https://arxiv.org/abs/1212.0901v2) by Y. Bengio et al., Section 3.5.
  - [Ilya Sutskever’s thesis](http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf) (pdf), section 7.2

### Batch Normalization
- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167) by S. Ioffe et al.
- [Why does batch normalization help?](https://www.quora.com/Why-does-batch-normalization-help)

### Regularization
- Dropout
  - [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf) by N. Srivastava et al.
- DropConnect
  - [Regularization of Neural Networks using DropConnect](https://cs.nyu.edu/~wanli/dropc/)

### Hyper-Parameter Optimization
- [Random Search for Hyper-Parameter Optimization](http://www.jmlr.org/papers/v13/bergstra12a.html) by J.Bergstra et al.
- [Systematic evaluation of CNN advances on the ImageNet](https://arxiv.org/abs/1606.02228) by D. Mishkin et al.
  - In this article, they mention about the mini-batch size dependency
- [Practical recommendations for gradient-based training of deep architectures](https://arxiv.org/abs/1206.5533) by Yoshua Bengio


## CNN
### Object Recognition
Famous Networks
- LeNet
  - [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) by Y. LeCun et al., 1998.
- AlexNet
  - [ImageNet Classification with Deep Convolutional Neural Networks](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) by Alex Krizhevsky et al., 2012.
- VGG
  - [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) by K. Simonyan et al., 2014.
- GoogLeNet
  - [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842) by C. Szegedy et al., 2014.
- Network In Network
  - [Network In Network](https://arxiv.org/abs/1312.4400) by M. Lin et al., 2014
    - This paper mentions about Global Average Pooling (GAP).
- ResNet
  - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) by K. He et al., 2015.
- Inception
  - [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567) by C. Szegedy et al., 2015.
  - [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261) by C. Szegedy et al., 2016.
  - blog post [Inception modules: explained and implemented](https://hacktilldawn.com/2016/09/25/inception-modules-explained-and-implemented/) by Tommy Mulc.
- Xception
  - [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357) by F. Chollet (Google), 2016.
- DenseNet
  - [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) by G. Huang et al., 2016.
- MobileNet
  - [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861) by A. G. Howard et al., 2017.
- SE Net
  - [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507) by Jie Hu et al., 2017.

Technics
- [Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results](https://arxiv.org/abs/1703.01780) by A. Tarvainen et al., 2017.
  
### Object Detection
Overview
- [Deep Learning for Generic Object Detection: A Survey](https://arxiv.org/abs/1809.02165) by Li Liu et al., 2018.  

Blogs  
- [Region of interest pooling explained](https://deepsense.ai/region-of-interest-pooling-explained/)  

Famous Networks
- Selective Search
  - [Selective Search for Object Recognition](https://staff.fnwi.uva.nl/th.gevers/pub/GeversIJCV2013.pdf) by J. R. R. Uijlings et al., 2013.
- R-CNN
  - [Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/abs/1311.2524) by R. Girshick et al., 2013.
- Fast R-CNN
  - [Fast R-CNN](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf) by R. Girshick et al.
- Faster R-CNN
  - [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497) by S. Ren et al., 2015.
- R-FCN
  - [R-FCN: Object Detection via Region-based Fully Convolutional Networks](https://arxiv.org/abs/1605.06409) by J. Dai et al., 2016.
- YOLO v1~v5
  - [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640) by J. Redmon et al., 2015~2016.
- SSD v1~v5
  - [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) by Wei Liu et al., 2015~2016.
- Mask RCNN v1~v3
  - [Mask R-CNN](https://arxiv.org/abs/1703.06870) by K. He et al., 2017~2018.
- RetinaNet v1~v2
  - [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) by Tsung-Yi Lin et al., 2017~2018.
  
### Segmentation
Famous Networks
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

### Pose Prediction
- Convolutional Pose Machines
  - [Convolutional Pose Machines](https://arxiv.org/abs/1602.00134) by Shih-En Wei et al., 2016.
- Open Pose
  - [Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields](https://arxiv.org/abs/1611.08050) by Zhe Cao et al., 2016.

### Other Information
#### Data Augmentation
- [Improved Regularization of Convolutional Neural Networks with Cutout](https://arxiv.org/abs/1708.04552) by T. DeVries et al., 2017.
- [Random Erasing Data Augmentation](https://arxiv.org/abs/1708.04896) by Z. Zhong et al., 2017.
- [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412) by H. Zhang et al., 2017.

#### Feature Visualization
- [Visualizing and Understanding Convolutional Networks](https://arxiv.org/abs/1311.2901) by Matthew D. Zeiler et al., 2013.
- Class activation maps
  - Learning Deep Features for Discriminative Localization [[Paper]](https://arxiv.org/abs/1512.04150) [[website]](http://cnnlocalization.csail.mit.edu/) by B. Zhou et al., 2015.
- [Striving for Simplicity: The All Convolutional Net](https://arxiv.org/abs/1412.6806) by 
Striving for Simplicity: The All Convolutional Net
Jost Tobias Springenberg et al., 2014.

#### Transfer Learning
- [How transferable are features in deep neural networks?](https://arxiv.org/abs/1411.1792)

#### Others
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) by K. Simonyan et al., 2014.
- [CNN Tricks](http://lamda.nju.edu.cn/weixs/project/CNNTricks/CNNTricks.html)
- [A guide to receptive field arithmetic for Convolutional Neural Networks](https://medium.com/@nikasa1889/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807)
- [A guide to convolution arithmetic for deep learning](https://arxiv.org/abs/1603.07285) by V. Dumoulin et al. 2016.


## RNN & LSTM

### RNN
- [TDNN (Time delay neural network) wiki](https://en.wikipedia.org/wiki/Recurrent_neural_network#Elman_networks_and_Jordan_networks)
- [Elman Network](http://onlinelibrary.wiley.com/doi/10.1207/s15516709cog1402_1/abstract) by Jeffrey L. Elman
- [Elman Network wiki](https://en.wikipedia.org/wiki/Recurrent_neural_network#Elman_networks_and_Jordan_networks)

### GRU
- [Learning Long-Term Dependencies with RNN](http://www.cs.toronto.edu/~guerzhoy/321/lec/W09/rnn_gated.pdf) 

### LSTM
- [LONG SHORT-TERM MEMORY](http://www.bioinf.jku.at/publications/older/2604.pdf) by Sepp Hochreiter and Jürgen Schmidhuber.
- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Exploring LSTMs](http://blog.echen.me/2017/05/30/exploring-lstms/)

### RNN Hyperparamters (model structures)
- LSTM Vs GRU
  - [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](https://arxiv.org/abs/1412.3555) by J. Chung et al.
  - [An Empirical Exploration of Recurrent Network Architectures](http://proceedings.mlr.press/v37/jozefowicz15.pdf) by R. Jozefowicz et al.
  - [Visualizing and Understanding Recurrent Networks](https://arxiv.org/abs/1506.02078) by A. Karpathy et al.
  - [Massive Exploration of Neural Machine Translation Architectures](https://arxiv.org/abs/1703.03906v2) by D. Britz et al.
  - [How to Generate a Good Word Embedding?](https://arxiv.org/abs/1507.05523) by S. Lai et al.
- Example RNN Architectures
  - [Neural Speech Recognizer: Acoustic-to-Word LSTM Model for Large Vocabulary Speech Recognition](https://arxiv.org/abs/1610.09975) by H. Soltau et al.
  - [Speech Recognition with Deep Recurrent Neural Networks](https://arxiv.org/abs/1303.5778) by A. Graves et al.
  - [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) by I. Sutskever et al.
  - [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555) by O. Vinyals et al.
  - [DRAW: A Recurrent Neural Network For Image Generation](https://arxiv.org/abs/1502.04623) by K. Gregor et al.
  - [A Long Short-Term Memory Model for Answer Sentence Selection in Question Answering](http://www.aclweb.org/anthology/P15-2116) by D. Wang et al.
  - [SEQUENCE-TO-SEQUENCE RNNS FOR TEXT SUMMARIZATION](https://pdfs.semanticscholar.org/3fbc/45152f20403266b02c4c2adab26fb367522d.pdf) by R. Nallapati et al., 2016.

### Word2Vec
  - [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) by T. Mikolov et al.
  - [Distributed Representations of Words and Phrases and their Compositionality](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) by T. Mikolov et al.
  - [Word2Vec Tutorial - The Skip-Gram Model](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)
  - [Word2Vec (Part 1): NLP With Deep Learning with Tensorflow (Skip-gram)](http://www.thushv.com/natural_language_processing/word2vec-part-1-nlp-with-deep-learning-with-tensorflow-skip-gram/)
  - [TensorFlow word2vec tutorial](https://www.tensorflow.org/tutorials/word2vec)

### Other Information
- [A Beginner’s Guide to Recurrent Networks and LSTMs](https://deeplearning4j.org/lstm.html)
- [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [Preprocessing text before use RNN](https://datascience.stackexchange.com/questions/11402/preprocessing-text-before-use-rnn)
- [Recurrent Batch Normalization](https://arxiv.org/abs/1603.09025)

## GAN
- [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) by Ian J. Goodfellow et al., 2014.
- [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434) by A. Radford et al., 2015.
- [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498) by T. Salimans et al., 2016.
- [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) by P. Isola et al., 2017.
- [NIPS 2016 Tutorial: Generative Adversarial Networks](https://arxiv.org/abs/1701.00160) by Ian Goodfellow.

## Reinforcement Learning
- [Producing flexible behaviours in simulated environments](https://deepmind.com/blog/producing-flexible-behaviours-simulated-environments/)
- [Emergence of Locomotion Behaviours in Rich Environments](https://arxiv.org/abs/1707.02286)
- [Technical Note Q-Learning](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.80.7501&rep=rep1&type=pdf)
- [A Theoretical and Empirical Analysis of Expected Sarsa](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.216.4144&rep=rep1&type=pdf)
- [Random Features for Large-Scale Kernel Machines](https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf) by Rahimi & Recht, 2007.
- [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)
  - Deep Deterministic Policy Gradients(DDPC) paper
- [Reinforcement learning for robots using neural networks.](https://pdfs.semanticscholar.org/54c4/cf3a8168c1b70f91cf78a3dc98b671935492.pdf) by Long-Ji Lin, 1993.
- [Human-level control through deep reinforcement learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) by V. Mnih et al., 2015.
- [Issues in Using Function Approximation for Reinforcement Learning (1993)](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.73.3097) by Sebastian Thrun , Anton Schwartz
- [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) by Hado van Hasselt et al., 2015.
- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952) by Schaul et al., 2016.
- [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581) by Z. Wang et al., 2015.
- [Deep Recurrent Q-Learning for Partially Observable MDPs](https://arxiv.org/abs/1507.06527) by Hausknecht & Stone, 2015.

## Tips for the real-world problem
- [Learning from Imbalanced Classes](https://svds.com/learning-imbalanced-classes/)
- [How to handle Imbalanced Classification Problems in machine learning?](https://www.analyticsvidhya.com/blog/2017/03/imbalanced-classification-problem/)
- [How to do imbalanced classification in deep learning (tensorflow, RNN)?](https://datascience.stackexchange.com/questions/17219/how-to-do-imbalanced-classification-in-deep-learning-tensorflow-rnn)
- [Learning Deep Representation for Imbalanced Classification](http://mmlab.ie.cuhk.edu.hk/projects/LMLE.html) by C. Huang et al., 2016.
- [A systematic study of the class imbalance problem in convolutional neural networks](https://arxiv.org/abs/1710.05381) by M. Buda et al., 2017.

## Websites & blogs

- Segmentation
  - [A 2017 Guide to Semantic Segmentation with Deep Learning](http://blog.qure.ai/notes/semantic-segmentation-deep-learning-review)

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
  - [Implementing Batch Normalization in Tensorflow](https://r2rt.com/implementing-batch-normalization-in-tensorflow.html)
- Music and Art
  - [Magenta](https://magenta.tensorflow.org/) is a research project exploring the role of machine learning in the process of creating art and music. 

- OpenAI Gym
  - [OpenAI Gym Beta](https://blog.openai.com/openai-gym-beta/)
  

## My Memo
- [Stanford Vision Lab.](http://vision.stanford.edu/publications.html)
- ISIC Challenge 2017
  - [SKIN LESION ANALYSIS TOWARD MELANOMA DETECTION: A CHALLENGE AT THE 2017 INTERNATIONAL SYMPOSIUM ON BIOMEDICAL IMAGING (ISBI), HOSTED BY THE INTERNATIONAL SKIN IMAGING COLLABORATION (ISIC)](https://arxiv.org/abs/1710.05006)
  - [Image Classification of Melanoma, Nevus and Seborrheic Keratosis by Deep Neural Network Ensemble](https://arxiv.org/abs/1703.03108)
  - [Incorporating the Knowledge of Dermatologists to Convolutional Neural Networks for the Diagnosis of Skin Lesions](https://arxiv.org/abs/1703.01976)
  - [RECOD Titans at ISIC Challenge 2017](https://arxiv.org/abs/1703.04819)
  
## Datasets or Competition
- [List of datasets for machine learning research (wikipedia)](https://en.wikipedia.org/wiki/List_of_datasets_for_machine_learning_research)
- [Kaggle](https://www.kaggle.com/)
- [Deep Analytics](https://deepanalytics.jp/?tc=menu)
