# RCNN-for-SCT
Use RCNN to solve story cloze test

三个版本：

- Glove + full story：输入的四个句子全都考虑 。 
- Glove + last sentence：输入的四个句子中只考虑最后一个句子。  
- **skip-thoughts + last sentence**：使用skip-thoughts vector，输入的四个句子中只考虑最后一个句子  

以skip-thoughts + last sentence为最终版本。  



##### 1.1 编译/运行环境  

- 开发及测试均在windows下进行  
- tensorflow-gpu 1.12.0  
- python 3.6.6  
- 所需库：pickle, tqdm, numpy



##### 1.2 数据说明  

- rawdata文件夹中的test,train和val    

- 词向量：  

  - Glove (6B tokens, 400K vocab, uncased, 50d, 100d, 200d, & 300d vectors, 822 MB)```(参考资料[2])```
  - 在RCNN的最终版本中使用了skip-thoughts vector```（参考资料[3]）```。skip-thoughts vector需要结合模型自行生成。  



##### 1.3 代码文件说明  

共9个python文件：

- preprocess.py（Glove的两个版本使用）：对项目给的数据集（train.csv, val.csv, test.csv）进行数据预处理。加载词向量文件（Glove），建立词典，建立原始数据对词典的索引。    
- data_util.py：集成了与数据处理相关的函数：
  - load_embedding:加载词向量（仅在Glove版本中用到）。
  - hold_out：将validation集分成两部分，一部分作训练集，一部分作测试集。
  - load_data（Glove + last sentence使用）: 加载数据，padding, 将数据处理为可以输入给RCNN的形式 。 
  - load_data_fullstory（Glove + full story使用）：功能同上，有微调。  
  - load_skipthoughts（skip-thoughts + last sentence使用）：直接把数据处理成向量形式输入给RCNN。
- BiLSTM_fullstory.py（用于Glove + full story）：BiLSTM的结构与loss计算。
- BiLSTM.py（用于Glove + last sentence）：BiLSTM的结构与loss计算。
- BiLSTM_skipthoughts.py（用于skip-thoughts + last sentence）：BiLSTM的结构与loss计算。
- RCNN_train_fullstory.py（用于Glove + full story）：进行训练和准确率评估，评估结果写入data/result文件夹中。 
- RCNN_train.py（用于Glove + last sentence）：进行训练和准确率评估，评估结果写入data/result文件夹中。  
- RCNN_train_skipthoughts.py（用于skip-thoughts + last sentence）：进行训练和准确率评估，评估结果写入data/result文件夹中。  
- get_embd.py：用于获取skip-thoughts vector。
    

 



##### 1.4 实验方法  

- 以双向LSTM为主，但在LSTM的基础上做了一点改进。这一改进是从```参考文献[4]```中得到启发的，如下图：

![](http://ww1.sinaimg.cn/mw690/0071tMo1ly1fyfgvryk3ej30t10ca0v2.jpg)

- 简略地说，就是在双向RNN（在这个项目中，将原始RNN替换为双向LSTM）的输出结果上再套一层max-pooling layer。从CNN的角度来看，即卷积层为BiLSTM，然后经过max-pooling layer，最后达到输出层，得出对(story, answer) pair的score。由于有两个备选句子，所以要分别得出(story, ans1)和(story, ans2)的score，取大者为正确答案，输出label。  

- 对于BiLSTM而言，训练时的输入是三元组（story, true answer, false answer）*batch size。这里是将验证集的一部分作为训练集的。之所以这样做，是因为我在阅读论文时发现story cloze test的训练集是有问题的，在很多模型上甚至只会起到负面作用，每一条数据只有正确答案，没有错误答案，机器无法明确自己的学习目的。在Glove + fullstory版本中， 输入的story是四个语境句子的句向量的叠加平均；在Glove + last sentence版本中，输入的story是最后一个语境句子的句向量；在skip-thoughts + last sentence版本中，输入的story是最后一个语境句子的句向量。

  max-pooling layer的输入是BiLSTM的输出，激活函数为tanh。  

- 采用SGD进行训练。  



在以上网络架构的基础上，使用max-margin loss来计算损失：

- 经过max-pooling layer之后，得到了story, true answer, false answer分别对应的结果（记为s, t, f），这三个结果都是向量形式。   

- 计算s和t的cosine similarity（记为$$cos_t$$）以及s和f的cosine similarity（记为$$cos_f$$）：  
  $$
  cos(s,t) = \frac {\vec{s} \cdot \vec{t}} {|s|\cdot|t|}
  $$

  $$
  cos(s,f) = \frac {\vec{s} \cdot \vec{f}} {|s|\cdot|f|}
  $$











- 使用max-margin计算loss：
  $$
  max-margin_{loss} = max(0, margin - cos_t + cos_f)
  $$
  margin预先设定。  

- 测试时的输入为（query, answer）二元组，模型为该二元组打分，打分公式为：
  $$
  score = cos(query, answer)
  $$
  也即query和answer的余弦相似度。

  在两个备选答案中判score高的那个答案为正确答案。    



##### 1.5 实验步骤  

进行训练：  

- 目标：最小化max-margin loss  
- 采用SGD进行训练：
  - Glove + full story：
    - 最大句子长度30，即句向量为9000维（一个词向量为300维）
    - margin: 0.2
    - rnn_size: 128
    - Dropout keep probability: 0.65
    - Learning_rate: 0.1
    - Learning rate down rate:0.6
    - Learning rate down times :5 
    - batch_size: 32
    - epoch: 10 （由于设定了每一个学习率对应10个epoch，共有5个不同的学习率，所以共50个epoch）  
  - Glove + last sentence:  
    - 最大句子长度30，即句向量为9000维（一个词向量为300维）
    - margin: 0.15
    - rnn_size: 128
    - Dropout keep probability: 0.65
    - Learning_rate: 0.1
    - Learning rate down rate:0.9
    - Learning rate down times :4 
    - batch_size: 32
    - epoch: 10 （由于设定了每一个学习率对应10个epoch，共有4个不同的学习率，所以共40个epoch）  
  - **skip-thoughts + last sentence**:  
    - 一个句向量维度为4800维
    - margin: 0.20
    - rnn_size: 512
    - Dropout keep probability: 0.70
    - Learning_rate: 0.1
    - Learning rate down rate:0.6
    - Learning rate down times :5
    - batch_size: 32
    - epoch: 10 （由于设定了每一个学习率对应10个epoch，共有5个不同的学习率，所以共50个epoch）  





##### 1.6 实验结果（在验证集上的准确率）

- RCNN + Glove + full story

  | epoch | accuracy           |
  | ----- | ------------------ |
  | 1     | 0.5722801788375559 |
  | 10    | 0.6095380029806259 |
  | 20    | 0.6408345752608048 |
  | 30    | 0.6527570789865872 |
  | 40    | 0.6631892697466468 |


- RCNN + Glove + last sentence  

  | epoch | accuracy           |
  | ----- | ------------------ |
  | 1     | 0.5663189269746647 |
  | 10    | 0.6333830104321908 |
  | 20    | 0.6959761549925484 |
  | 30    | 0.7019374068554396 |
  | 40    | 0.698956780923994  |


- **RCNN + skip-thoughts + last sentence** 

  | epoch | accuracy               |
  | ----- | ---------------------- |
  | 1     | 0.6199460916442049     |
  | 10    | 0.7304582210242587     |
  | 20    | 0.738544474393531      |
  | 30    | **0.7520215633423181** |
  | 40    | 0.7331536388140162     |

由以上结果可知，只考虑最后一个语境句子效果要好于考虑全部4个句子；同时，由于skip-thoughts在句向量的表示上更占优势，将skip-thoughts和last sentence结合起来可以达到更好的效果。  
