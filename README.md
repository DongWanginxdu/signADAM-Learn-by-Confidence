# signADAM: Learning Confidences for Deep Neural Networks
## introduction
The code for signADAM: Learning Confidences for Deep Neural Networks. https://arxiv.org/abs/1907.09008<br>
Based the following two motivativations：<br>
- Hard and easy samples both have easy features. If the neural networks continue learning these easy features, the neural networks tend to overfit. So easy features should be inhibitted. 
- Neurons tend to have sparse activation.


We use graients to measure the speed of feature learning. So a confidence with zero can exactly satify the above motivations. <br>
![Image text](https://github.com/DongWanginxdu/signADAM-Learn-by-Confidence/blob/master/img/img.pdf)
