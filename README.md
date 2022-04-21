# signADAM: Learning Confidences for Deep Neural Networks
## Introduction
The code for signADAM: Learning Confidences for Deep Neural Networks.

 [Arxiv](https://arxiv.org/abs/1907.09008)   [IEEE]( [signADAM++: Learning Confidences for Deep Neural Networks | IEEE Conference Publication | IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/8955622) )

Based the following two motivations：<br>

- Hard and easy samples both have easy features. If the neural networks continue learning these easy features, the neural networks tend to overfit. So easy features should be inhibited. 
- Neurons tend to have sparse activation.

We use gradients to measure the speed of feature learning. So a confidence with zero can exactly satisfy the above motivations. <br>
![image](C:\Users\wd\Desktop\r4t5t\signADAM-Learn-by-Confidence\img\show.PNG)

## News!

This paper has been accepted by  2019 International Conference on Data Mining Workshops (ICDMW) .

A 15-min oral presentation has been given in Beijing.

Our signADAM++ algorithm is used in various  Deep Neural Network Training Processors, such as:

[HNPU: An Adaptive DNN Training Processor Utilizing Stochastic Dynamic Fixed-Point and Active Bit-Precision Searching]( [HNPU: An Adaptive DNN Training Processor Utilizing Stochastic Dynamic Fixed-Point and Active Bit-Precision Searching | IEEE Journals & Magazine | IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/9383824) )

[An Energy-Efficient Deep Neural Network Training Processor with Bit-Slice-Level Reconfigurability and Sparsity Exploitation]( [An Energy-Efficient Deep Neural Network Training Processor with Bit-Slice-Level Reconfigurability and Sparsity Exploitation | IEEE Conference Publication | IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/9410324) )

[A Mobile DNN Training Processor With Automatic Bit Precision Search and Fine-Grained Sparsity Exploitation]( [A Mobile DNN Training Processor With Automatic Bit Precision Search and Fine-Grained Sparsity Exploitation | IEEE Journals & Magazine | IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/9650747) )

## Quick Start
```python
cd CIFAR-classification
bash run.sh
```
## Requirments

Our source code heavily relies on the [repository]( [kuangliu/pytorch-cifar: 95.47% on CIFAR10 with PyTorch (github.com)](https://github.com/kuangliu/pytorch-cifar) )

```
PyTorch >= 1.0
Python >= 3.6
```

# Usage

```python
from algorithms.signadam import *
optimizer = SIGNADAMP(net.parameters(), lr=args.lr, threshold = args.th, weight_decay=5e-4)
```



# Citation

If you find our method is valuable, please cite as follows.

```
@inproceedings{wang2019signadam++,
  title={SignADAM++: Learning confidences for deep neural networks},
  author={Wang, Dong and Liu, Yicheng and Tang, Wenwo and Shang, Fanhua and Liu, Hongying and Sun, Qigong and Jiao, Licheng},
  booktitle={2019 International Conference on Data Mining Workshops (ICDMW)},
  pages={186--195},
  year={2019},
  organization={IEEE}
}
```

