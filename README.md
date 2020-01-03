##### Please Note That: This study is my own master's dissertation and all the copyrights belong to me. Please cite when you use in order to avoid any plagiarism.



# Deep Learning Implementations on Acoustic Wave Processing with LSTM Architecture



## ABSTRACT


Recently, deep learning has become one of the most robust, flexible, accurate and reliable tools used in almost every field. The contribution of Google’s TensorFlow machine learning platform in this new era is incontestable. By the decision of Google for making the TensorFlow platform open-source, a new perception started to spread out not only among mathematicians and computer scientists but also among entrepreneurs and large-scaled companies from all sectors.

Even though the deep learning methods are capable of handling almost any type of problem, their efficiency in finding proper solutions and the appropriateness of such methods to specific case studies should be well considered before deploying. This is due to the risk of their complicated tensor structures as well as due to the demanding calculation requirements of deep learning mechanisms. Therefore, before deciding to implement a deep learning model to business, it is advisable to investigate whether simpler solutions are available or whether deep learning support is indeed required. 

This research focuses on analysing whether it is feasible to deploy deep learning for waveforms analysis and slowness prediction in the petroleum industry. The dataset used for this research was provided by Weatherford which is one of the pioneer companies in the oil and natural gas services field. 

For this project, it was started by assessing the potential of getting the same results from a simplified version of the current industry standard. To do so, it was determined to work with extreme points and the global maximum and local minimum was suggested. Results showed that this approach was not appropriate for this case study due to inconsistent waveform structure.

Following this attempt, another deep learning model was built using LSTM architecture and consulting Google’s TensorFlow library. This model was trained in different wells’ data. The results suggested that, on one hand, the LSTM model is capable to learn and later predict the results similar to what it has already learned. On the other hand, this model results in completely different findings when requested to use data it has never encountered before.

Consequently, LSTM architecture is beneficial in terms of it is suitable for handling tensors. It results also show that as the model gets trained, it is more likely to get more accurate results.
