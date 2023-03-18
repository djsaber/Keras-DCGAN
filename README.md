# Keras-DCGAN
基于Keras搭建一个深度卷积生成对抗网络DCGAN，用动漫头像数据集对DCGAN进行训练，完成模型的保存和加载和生成测试。

环境：<br />
CUDA：11.6.134<br />
cuDNN：8.4.0<br />
keras：2.9.0<br />
tensorflow：2.9.1<br /><br />

注意：<br />
项目内目录中两个文件夹：<br />
1. /datasets：将数据集文件解压至此<br />
2. /save_models：保存训练好的模型权重文件，包括生成器权重和判别器权重两个文件<br />
3. /images：保存生成的样本<br /><br />

DCGAN概述<br />
2016年，Alec等人发表的论文《UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS》（简称DCGAN），
首次提出将CNN应用到GAN生成对抗网络的模型中，从而代替全连接层。这篇论文中讨论了GAN特征的可视化、潜在空间差值等。<br /><br />

DCGAN是GAN的一个变体，DCGAN就是将CNN和原始的GAN结合到一起，生成网络和鉴别网络都运用到了深度卷积神经网络。<br />
• 早期的GAN在图像上仅局限MNIST这样的简单数据集中，DCGAN使GAN在图像生成任务上的效果大大提升。<br />
• DCGAN几乎奠定了GAN的标准架构，之后GAN的研究者们不用再过多关注模型架构和稳定性，可以把更多的精力放在任务本身上。<br />

数据集：<br />
anime-faces：包含21,551张动漫头像图片，尺寸为64*64*3<br />
链接：https://pan.baidu.com/s/1IlqGfVOSXoVlFAC9zOxNfQ?pwd=52dl 提取码：52dl<br /><br />

每个eopch的step中的训练过程如下：<br />
训练K步判别器：<br />
1. 随机采样batch size个真实样本<br />
2. 随机采样batch size个随机噪声，输入生成器获得相等数量的生成样本<br />
3. 将真实样本和生成样本输入判别器进行训练，真实样本标签为1，生成样本标签为0<br />
训练V步生成器：<br />
4. 将生成器和判别器连接起来，并冻结判别器参数，使其不参与更新<br />
5. 随机采样batch size个随机噪声，输入连接起来的模型进行训练，标签为1<br /><br />

采用2:1训练比，每个steop判别器训练2次，生成器训练1次，根据不同场景可以更改此项。<br />
需要注意的是，Keras训练过程中模型的状态依据的是model.compile()时的状态。<br />
在定义判别器后，通过compile()编译判别器，然后连接生成器和判别器构成combined模型，并将判别器的trainable设置为False，再通过compile()编译combined模型。<br />
在训练时判别器和combined模型各自保持编译时的状态，所以在训练过程中不需要反复将判别的的trainable参数设置为True和False<br />

代码中设置：<br />
数据归一化到-1到1之间<br />
生成器输出层采用tanh激活函数<br />
判别器输出层采用softmax激活函数，输出一维向量<br />
采用label smooth平滑标签，clip=0.1<br />
使用adam优化器，lr=0.0002，beta=0.5<br />
使用crossentropy交叉熵损失函数<br />
生成器使用BN，momentum=0.8，判别器不用<br />
使用leakyrelu，激活函数，alpha=0.2<br />
生成器使用逆卷积strides=2上采样<br />
判别器使用strides=2卷积下采样<br />
生成器输入采用随机高斯分布噪声<br /><br />

GAN模式崩溃解决方法：<br />
1、使用合适的损失函数，有助于防止模式崩溃和收敛<br />
2、提供足够的潜在空间，潜在空间是对生成器的输入（随机噪声）进行采样<br />
3、保持安全的学习率，高学习率会导致导致模式崩溃或不收敛<br />
4、特征匹配，特征匹配提出了一个新的目标函数，不直接使用鉴别器输出。生成器经过训练，使得生成器的输出预期与鉴别器的中间特征上的真实图像值相匹配。<br />
5、使用历史平均值，在处理非凸目标函数时，历史平均可以帮助收敛模型<br />
（生成图片缓存池）缓存生成图像，每次只取当前batch的一半，从缓存池里取剩下的一半，用于接下来的判别，提高判别器的稳定性。<br /><br />

GAN训练trick：<br />
1、输入规范化为-1到1之间<br />
2、修改损失函数<br />
3、使用标准正态分布的随机噪声<br />
4、一个mini-batch里面必须保证只有Real样本或者Fake样本，不要把它们混起来训练<br />
5、如果不能用batch norm，可以用instance norm<br />
6、避免稀疏梯度：ReLU, MaxPool<br />
  对于下采样，使用：Average Pooling，Conv2d + stride<br />
  对于上采样，使用：PixelShuffle, ConvTranspose2d + stride<br />
7、使用 Gradient Penalty<br />
8、如果还有可用的类别标签，在训练D判别真伪的同时对样本进行分类<br />
9、给输入增加噪声，随时间衰减<br />
10、多训练判别器D<br />
11、使用VAE的decoder权重来初始化generator<br />
12、软标签和带噪声的标签<br />
13、反转标签(Real=False, fake=True)<br />
14、监控梯度变化<br />



