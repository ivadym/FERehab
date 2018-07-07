# FER
## Facial Expression Recognition based on Convolutional Neural Networks and Transfer learning
 
In this project, developed in **Python 2.7.2** and using the **Keras API**, the fine-tuning was carried out on the **Google Cloud Platform** of the Inception-v3, Inception-ResNet-v2 and ResNet-50 models employing the **FER-2013** database. The reported results, which can be consulted in the following table, have been obtained on the private test set of the FER-2013 database.

|  Fine-tuned models  |                       Initial weights                      | Accuracy | Duration of training  (NVIDIA Tesla P100) | Number of Parameters | Size of the models |
|:-------------------:|:----------------------------------------------------------:|:--------:|:-----------------------------------------:|:--------------------:|:------------------:|
|      ResNet-50      | [VGGFace2](http://www.robots.ox.ac.uk/~vgg/data/vgg_face/) |  71.25%  |           2h 29m 45s (40 epochs)          |      25,613,383      |      308.3 MB      |
| Inception-ResNet-v2 |            [ImageNet](http://www.image-net.org/)           |  65.00%  |           1h 4m 26s (27 epochs)           |      55,857,255      |      449.3 MB      |
|     Inception-v3    |            [ImageNet](http://www.image-net.org/)           |  63.86%  |           5h 9m 37s (80 epochs)           |      23,873,703      |      192.1 MB      |

### Directory index

- **src**. It is composed of the Python Notebooks used to describe the models and deploy them in the Google Cloud Platform.

- **FER-2013**. It contains the set of images from the FER-2013 database used for training, validation and evaluation. This division follows the patterns stipulated in the challenge [FER-2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data).

- **trained_models**. It contains the fine-tuned models that have allowed to obtain the rates shown in the table above.

- **test**. It contains a file to reproduce the evaluation made on the FER-2013 datasets and a script that recognizes the expressions in real time using the tools provided by OpenCV and the trained models.

- **CycleGAN**. It consists of scripts that try to reproduce a CycleGAN network to increase the number of images labeled with the expression of _disgust_ from the images of the _neutral_ class. The architecture is based on the [original implementation](https://github.com/junyanz/CycleGAN) in TensorFlow and inspired by the publication of [Xinyue Zhu et al.](https://arxiv.org/abs/1711.00648).

- **SmartMirror**. It contains a very simple interface developed to be used by a smart mirror. It also poses a game to the user who has to imitate the indicated facial expressions. The hits and failures, as well as the response time, are fed back to the user in real time. This development was made with **Python 3.6.4**.

### Demos (FER in real time and the game interface)
<div align='left'>
  <img src='demos/emotions.gif' width='400px'>
</div>

<div align='right'>
  <img src='demos/interface.gif' width='433px'>
</div>
