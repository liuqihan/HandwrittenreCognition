# HandwrittenreCognition
Recognition of Chinese Characters from Input Pictures
# 汉字书法识别自由练习赛

时间：2018/10/20-2018/10/21

地点：CSDN总部

https://www.tinymind.cn/competitions/41

### 数据集

竞赛数据提供100个汉字书法单字，包括碑帖，手写书法，古汉字等等。图片全部为单通道灰度jpg，宽高不定。

#### 训练集：

训练集每个汉字400张图片，共计40000张图片，供参赛人员测试和开发参赛算法模型，训练集是标注好的数据，图片按照图片上的文字分类到不同的文件夹中，也就是说文件夹的名字就是文件夹里所有图片的标签。

#### 测试集分两部分：

第一部分每汉字100张图片共10000张图片，在竞赛过程中，开放数据下载但不提供标签。比赛中第一阶段的排行榜对应参赛队伍第一部分数据的评测得分，这部分得分和排名不影响比赛的最终成绩，其目的是供参赛人员测试算法模型。

第二部分测试数据每汉字50张以上图片（单字图片数不固定）共16343张图片，比赛的最后阶段公开下载，不提供标签。

#### 数据下载地址：

训练集：链接: <https://pan.baidu.com/s/1UxvN7nVpa0cuY1A-0B8gjg> 密码: aujd

测试集: <https://pan.baidu.com/s/1tzMYlrNY4XeMadipLCPzTw> 密码: 4y9k

### 项目流程：

1、生成tfrecord数据；

2、加载预训练模型进行训练模型，保存ckpt文件；

3、模型调参；

4、校验集预测，生成csv文件。

### 项目中出现的问题汇总：

1、  数据预处理时使用inception_processing.py，并将其文件中的数据裁切和随机翻转去掉，因为汉字经过此过程后可能变成其他字体；

2、  模型训练时loss值偏大，达到50左右，其原因为weightdecay（0.004）偏大，默认值为0.00004，修改后loss值正常；

3、  训练模型时，validation结果曾出现非常差，正确率仅为0.01，其原因可能为获取checkpoint文件失败；重新生成tfrecord文件运行后得到了正常的结果；

4、生成提交的CSV文件时产生了3种思路：

（1）使用test数据生成tfrecord文件，利用原有框架生成识别结果文件；

（2）参考eval_image_classifier.py文件进行修改，读取test图片，预处理，进行预测，最终生成预测文件；

（3）利用ckpt文件生成pb文件，进而生成预测文件；

5、对test数据进行预测时，获取checkpoint时出现问题，因为指定的路径内没有checkpoint文件，无法获取最近的运行结果，修改后直接restore即可

6、提交数据的格式出现问题，label按列表的格式进行了保存，与比赛的要求不符，修改后正常。

### 体会：

1、  项目中遇到问题时，先独立思考，再执行666法则，及时与同学及老师交流；

2、  队员之间相互分享遇到的问题及解决办法。

### 命令行：

#### 生成数据TFRecord数据文件

cd /home/dl/offline/quiz-w7-2
python download_and_convert_data.py \
​    --dataset_name=quiz \
​    --dataset_dir=/home/dl/offline/comp.jpg/train

#### 训练模型（服务器）

python train_image_classifier.py \
  --dataset_dir=/home/dl/offline/comp.jpg/train \
  --dataset_name=quiz \
  --train_dir=/home/dl/offline/quiz-model/ckpt \
  --dataset_split_name=train \
  --checkpoint_path=/home/dl/offline/inception_v4/inception_v4.ckpt \
  --checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits/Aux_logits \
  --model_name=inception_v4 \
  --preprocessing_name=inception \
  --max_number_of_steps=25000 \
  --batch_size=32 \
  --save_interval_secs=300 \
  --save_summaries_secs=300 \
  --log_every_n_steps=100 \
  --optimizer=rmsprop \
  --learning_rate=0.05 \
  --learning_rate_decay_factor=0.1 \
  --num_epochs_per_decay=0.5 \
  --weight_decay=0.0001

#### 验证eval数据

python eval_image_classifier.py \
  --checkpoint_path=/home/sophiayu/Models/train_ckpt \
  --eval_dir=~ \
  --dataset_name=quiz \
  --dataset_split_name=validation \
  --dataset_dir=/home/sophiayu/Data/手写汉字识别数据/train \
  --model_name=inception_v4 \
  --preprocessing_name=inception \
  --batch_size=32 \
  --max_num_batches=128

#### 测试

python eval_image_classifier_quiz.py \
  --checkpoint_path=/home/sophiayu/Models/train_ckpt/model.ckpt-9914 \
  --eval_dir=~ \
  --dataset_name=quiz \
  --dataset_split_name=test \
  --dataset_dir=/home/sophiayu/Data/手写汉字识别数据/test2 \
  --model_name=inception_v4 \
  --preprocessing_name=inception \
  --batch_size=32 \
  --max_num_batches=128

#### 创建tensorboard 

tensorboard --logdir=/home/dl/offline/quiz-model/train_ckpt



