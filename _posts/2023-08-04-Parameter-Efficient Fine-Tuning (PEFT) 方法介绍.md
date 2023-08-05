## Parameter-Efficient Fine-Tuning
* * *
## 1. 前言
参数有效的调优(Parameter-Efficient Fine-Tuning, peft) 是为了高效的适配预训练的大模型的一类方法，其主要目的是在不调优预训练大模型参数的基础上，让大模型具备特定任务知识的能力。

如果想要一个深度模型具备我们期望的某种能力，比如给定关键词自动生成广告语，我们通常的做法是：
 - **完全监督学习范式**，在新的数据集上从零开始训练模型，让模型具有相关的能力。也就是选择一个恰当的模型，使用人工标注好的**广告语生成**数据从零开始训练模型，并设置合适的损失函数和验证数据来验证模型的有效性。这是传统NLP处理的方法，适合模型参数较小的情况下。并且，这类算法是case-by-case训练的，即换了一个新的任务就需要使用适合新任务的标注数据重新训练模型，**可扩展性**不够。
 - **预训练，精调范式 (Pre-train, Fine-tune)**，使用预训练（pretrained）的模型为基础，在新的数据集上fine-tuning，使得模型具有新的技能。预训练模型在前几年取得了很大的进展和效果，并伴随**迁移学习**的应用，成为了计算机视觉和自然语言处理模型训练的标准做法，即使用对应任务的**较小规模**的数据，对预训练模型进行调优，把预训练模型作为模型训练的具有足够知识的初始模型。迁移学习+预训练使得模型训练能够站在**巨人**的肩膀上前行，有效的降低了训练数据的规模和模型初始化的难度。
 - **预训练，提示，预测范式（Pre-train, Prompt, Predict）**， 使用参数有效的调优策略（Parameter-Efficient Fine-Tuning， PEFT）给预训练模型打补丁，以补丁的形式让模型具备新的技能。这是最近几年伴随着大模型的流行发展起来的新的研究热点。以GPT3为例，它有多达1750 亿的参数，即使使用FP16 来储存模型，也需要350 GBs 的GPU 显存，即至少5块需要80G显存的A100显卡才能把GPT3模型运行起来，如果要使用上述的fine-tuning策略，则至少需要10块80G显存的A100才能以很小的batch size进行调优。PEFT算法另辟蹊径，采用给大模型**打补丁**的方式，给已经训练好的大模型**注入新的知识**。一般来说，PEFT算法需要训练的参数量只有大模型参数量的0.1%～2%左右，对于像ChatGLM-6B这样的大模型，只需要一块显存12Gb的显卡就可以进行调优，极大降低了大模型的使用难度。

本文主要讲主要介绍当前最流行的几种PEFT算法，我们将从算法的基本思想开始，逐个分析以下几个PEFT算法，最后对各个算法的优缺点进行分析和对比。后续会介绍到的PEFT算法如下：
- LoRA: [LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS](https://arxiv.org/abs/2106.09685)
- Prefix Tuning: [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://aclanthology.org/2021.acl-long.353/), [P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks](https://arxiv.org/pdf/2110.07602.pdf)
- P-Tuning: [GPT Understands, Too](https://arxiv.org/abs/2103.10385)
- Prompt Tuning: [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/abs/2104.08691)
- AdaLoRA: [Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2303.10512)
* * *
## 2.  Prompt介绍
### 2.1 提出Prompt的动机
随着当前语言模型（Language Models）越来越大，Fine-tune的成本也越来越高。如前文所述，如果要调优GPT3至少需要10块80G显存的A100才能以很小的batch size进行调优，显然效率和成本都是不现实的。因此，GPT3没有用到Fine-tune，而是直接使用Prompt来完成任务，即采用如下方式完成将cheese这个词翻译成法语：
```
	Translate English to French: /* task description */
	cheese => ___ /* prompt */
```
上面的例子是zero-shot learning，我们再来看看one-shot learning：
```
Translate English to French: /* task description */
sea otter => loutre de mer /* example */
cheese => ___ /* prompt */
```
few-shot learning 的原理完全一致，就是把example多写几个，一行一个即可。事实证明，用few-shot，100个example就能达到非常好的性能。

相比于fine-tuning(如下图)，每次输入一个新的example就需要对整个大模型进行梯度更新。这样做一方面需要保存大模型参数的同时，还需要保存对应的梯度，以及训练状态，模型越大需要的硬件资源就越多，所需的训练数据量也就越大；另一方面，由于当前大模型已经是训练好了的，即它在其训练数据所在的领域已经具备了很好的性能，在新的任务上做fine-tuning的时候，会破坏其在原来领域已经具备的能力，即模型会出现**遗忘**。
![4cb68baf50514babbc237931e2b401f3.png](/images/4cb68baf50514babbc237931e2b401f3.png)

对于大语言模型来说，其本身在训练时已经学习到了大量的文本语料和任务，可以直接使用Prompt来完成新的任务。

总结一下，对于GPT3和ChatGPT，使用Prompt的根本方法是：
```
自然语言指令（task description） + 任务示例（example） + 带"__"的任务。
```
其中，**自然语言指令**和**任务示例**可以省略。**需要说明的是**，Prompt描述的**越精确越详细**，大模型所表现出来的性能就**越好**。

### 2.2 PEFT算法
#### 2.2.1 硬提示(hard prompt)
上文2.1节讲述的Prompt属于**硬提示(hard prompt)**，即需要模型在这个域上有比较多的经验，并且在使用前需要使用者知道这个模型的底层是什么样的。否则，**硬提示的性能一般会比Fine-tuning的结果差很多**。据研究硬提示有两个性质：
- 人类认为不错的硬提示对于LM来说不一定是一个好的硬提示，这个性质被称为硬提示的sub-optimal（次优）性。
- 硬提示的选择对于预训练模型的影响非常大。

如下图的完形填空测试，对于同一个任务，不同的**硬提示**对任务的准确率的变动很大，最好的top1准确率可以到51.08%，而最差的只有19.78%，第一行为训练时的表述方式。
![be7a2060f225fa13f39e230a419eee2c.png](/images/be7a2060f225fa13f39e230a419eee2c.png)

由此可见，人类认为的最好的Prompt，并不是大语言模型认为的最好的Prompt，人类和大语言模型之间存在着某种未知的鸿沟。于是，为了弥补这个鸿沟，需要不断尝试人工设计的**硬提示**，并记录和总结大语言模型的输出结果，不断在试错中积累经验。

#### 2.2.2 软提示(soft prompt)
就是因为**硬提示**存在这样的问题，2020年，科学家提出了软提示。软提示把Prompt的生成本身作为一个任务进行学习，相当于把Prompt的生成从人类一个一个尝试（离散）变换成机器自己进行学习和连续尝试。

由于需要机器自己学习，**软提示**不可避免地往模型内引入了新的参数。于是，如何**参数有效**地学习软提示？

#### 2.2.3 参数有效的调优策略（PEFT）算法
为了解决大型预训练模型的训练调优和人工设计Prompt的困难，研究人员开始研究参数有效的调优策略（Parameter-Efficient Fine-Tuning，PEFT）技术。PEFT技术旨在通过最小化微调参数的数量和计算复杂度，来提高预训练模型在新任务上的性能，从而缓解大型预训练模型的训练成本。这样一来，即使计算资源受限，也可以利用预训练模型的知识来迅速适应新任务，实现高效的迁移学习。因此，PEFT技术可以在提高模型效果的同时，大大缩短模型训练时间和计算成本，让更多人能够参与到深度学习研究中来。下面我们将深入探讨PEFT的一些主要做法。

**Adapter Tuning**
谷歌的研究人员首次在论文《[Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/pdf/1902.00751.pdf)》提出了针对BERT的PEFT微调方式，拉开了PEFT研究的序幕。他们指出，在面对特定的下游任务时，如果进行Full-fintuning（即预训练模型中的所有参数都进行微调），太过低效；而如果采用固定预训练模型的某些层，只微调接近下游任务的那几层参数，又难以达到较好的效果。

于是他们设计了如下图所示的Adapter结构，将其嵌入Transformer的结构里面，在训练时，固定住原来预训练模型的参数不变，只对新增的Adapter结构进行微调。同时为了保证训练的高效性（也就是尽可能少的引入更多参数），他们将Adapter设计为这样的结构：首先是一个down-project层将高维度特征映射到低维特征，然后过一个非线形层之后，再用一个up-project结构将低维特征映射回原来的高维特征；同时也设计了skip-connection结构，确保了在最差的情况下能够退化为identity。
![acf6970ad999910b4602c9cc98ac7892.png](/images/acf6970ad999910b4602c9cc98ac7892.png)

从实验结果来看，该方法能够在只额外对增加的3.6%参数规模（相比原来预训练模型的参数量）的情况下取得和Full-finetuning接近的效果。

**Prefix Tuning**
Prefix Tuning方法由斯坦福的研究人员提出，与Full-finetuning更新所有参数的方式不同，该方法是在输入的token之前构造一段任务相关的virtual tokens作为Prefix，然后训练的时候只更新Prefix部分的参数，而模型中的其他部分参数固定。该方法其实和构造Prompt类似，只是Prompt是人为构造的“显式”的提示,并且无法更新参数，而Prefix则是可以学习的“隐式”的提示。这个方法简单来说就是在输入的prompt之前加一个学习到的前缀。
Prefix Tuning与全量fine-tuning的差异如下图所示，全量fine-tuning是对模型的每个参数都进行了更新，而Prefix Tuning只是在输入的Input tokens前面加上一些学习到的任务特定的virtual tokens。实验结果也说明了Prefix Tuning的方式可以取得不错的效果。但是，不同的任务前面需要添加的前缀virtual tokens的大小是不同的。
![30c1b8d7f03d1287758e86e91af1c8ca.png](/images/30c1b8d7f03d1287758e86e91af1c8ca.png)

**Prompt Tuning**
该方法可以看作是Prefix Tuning的简化版本，只在输入层加入prompt tokens，并不需要加入MLP进行调整来解决难训练的问题，主要在T5预训练模型上做实验。似乎只要预训练模型足够强大，其他的一切都不是问题。作者也做实验说明随着预训练模型参数量的增加，Prompt Tuning的方法会逼近Fine-tune的结果。
![58ca9d815a54786ac246b71fca4c265e.png](/images/58ca9d815a54786ac246b71fca4c265e.png)
作者在论文中展示的结果是：随着预训练模型参数的增加，一切的问题都不是问题，最简单的设置也能达到极好的效果。
- Prompt长度影响：模型参数达到一定量级时，Prompt长度为1也能达到不错的效果，Prompt长度为20就能达到极好效果。
- Prompt初始化方式影响：Random Uniform方式明显弱于其他两种，但是当模型参数达到一定量级，这种差异也不复存在。
- 预训练的方式：LM Adaptation的方式效果好，但是当模型达到一定规模，差异又几乎没有了。
- 微调步数影响：模型参数较小时，步数越多，效果越好。同样随着模型参数达到一定规模，zero shot也能取得不错效果。

**P-Tuning V2**
因为P-Tuning V2与Prefix Tuning的思路相似，所以先讲P-Tuning V2。P-Tuning V2的基本思想如下图所示，它与Prefix Tuning的区别在于，Prefix Tuning之时在输入的时候加上任务特定的virtual tokens，而P-Tuning V2是在模型的每一层的输入处都加上一个任务特定的virtual tokens。
![682569f2f9f4502a83c742e3e0a0317b.png](/images/682569f2f9f4502a83c742e3e0a0317b.png)

**P-Tuning**
P-Tuning的思路与Prefix Tuning的类似，但两者的区别在于：
- Prefix Tuning是将额外的embedding加在开头，看起来更像是模仿Instruction指令；而P-Tuning的位置则不固定，在前缀和中间位置都有可能。
- Prefix Tuning通过在每个Attention层都加入Prefix Embedding来增加额外的参数，通过MLP来初始化；而P-Tuning只是在输入的时候加入Embedding，并通过LSTM+MLP来初始化。

![1a7247fc9c722ca6ababa15c60163c44.png](/images/1a7247fc9c722ca6ababa15c60163c44.png)

**LoRA**
微软和CMU的研究者指出，现有的一些PEFT的方法还存在这样一些问题：
- 由于增加了模型的深度从而额外增加了模型推理的延时，如Adapter方法;
- Prompt较难训练，同时减少了模型的可用序列长度，如Prompt Tuning、Prefix Tuning、P-Tuning方法; 因为模型的输入长度是有限制的，这些方法都是需要在输入时加入额外的token。
- 往往效率和质量不可兼得，效果略差于full-finetuning。

有研究者发现：语言模型虽然参数众多，但是起到关键作用的还是其中低秩的本质维度（low instrisic dimension）。受到该观点的启发，提出了Low-Rank Adaption(LoRA)，设计了如下所示的结构，在涉及到矩阵相乘的模块，引入A、B这样两个低秩矩阵模块去模拟Full-finetune的过程，相当于只对语言模型中起关键作用的低秩本质维度进行更新。
![7fd3358de8cce0cdf14d5f0d42df4e14.png](/images/7fd3358de8cce0cdf14d5f0d42df4e14.png)
![e727b1ef88428f4a9d58cfe709fc12f8.png](/images/e727b1ef88428f4a9d58cfe709fc12f8.png)

这么做就能完美解决以上存在的3个问题：
- 相比于原始的Adapter方法“额外”增加网络深度，必然会带来推理过程额外的延迟，该方法可以在推理阶段直接用训练好的A、B矩阵参数与原预训练模型的参数相加去替换原有预训练模型的参数，这样的话推理过程就相当于和Full-finetune一样，没有额外的计算量，从而不会带来性能的损失。
- 由于没有使用Prompt方式，自然不会存在Prompt方法带来的一系列问题。
- 该方法由于实际上相当于是用LoRA去模拟Full-finetune的过程，几乎不会带来任何训练效果的损失，后续的实验结果也证明了这一点。

**AdaLoRA**
AdaLoRA是对LoRA的一种改进，它根据重要性评分动态分配参数预算给权重矩阵。具体做法如下：
- 调整增量矩分配。AdaLoRA将关键的增量矩阵分配高秩以捕捉更精细和任务特定的信息，而将较不重要的矩阵的秩降低，以防止过拟合并节省计算预算。
- 以奇异值分解的形式对增量更新进行参数化，并根据重要性指标裁剪掉不重要的奇异值，同时保留奇异向量。由于对一个大矩阵进行精确SVD分解的计算消耗非常大，这种方法通过减少它们的参数预算来加速计算，同时，保留未来恢复的可能性并稳定训练。

它和LoRA的差异如下图，上面的公式是AdaLoRA的，下面的公式是LoRA的。
![b5057709152381535a8645f0858b427b.png](/images/b5057709152381535a8645f0858b427b.png)

**QLoRA**
[QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/pdf/2305.14314.pdf)是一种高效的、可减少内存/显存使用量的微调方法，它能够在单个 48GB GPU 上微调 65B 参数量的大语言模型，同时可保留完整的 16 位微调任务性能。 QLORA 是一种量化的LoRA，它通过将LoRA的梯度量化到4比特来进一步降低内存/显存的使用量，而不影响LoRA的性能，其主要思想如下图所示。
![dd79075bbd83e2ed4bdf6e5c34c23377.png](/images/dd79075bbd83e2ed4bdf6e5c34c23377.png)

### 2.3 总结
huggingface开源了一个[peft](https://github.com/huggingface/peft)库，该库由Python语言实现，包括了当前最流行的几种PEFT策略算法。
在选择上述的PEFT算法时，由以下几点建议：
 - 当微调的参数量较多时，从结果来看，对FFN层进行修改更好。一种可能的解释是FFN层学到的是任务相关的文本模式，而Attention层学到的是成对的位置交叉关系，针对新任务并不需要进行大规模调整。
 - 当微调参数量较少（0.1%）时，对Attention进行调整效果更好。
* * *
