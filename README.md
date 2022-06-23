## 代码说明

### 环境配置

- Python 版本：3.8.10

- PyTorch 版本：1.7.0

- CUDA 版本：cu110
- 显存要求 >= 46G

python具体所需环境在 `requirements.txt` 中定义

### 数据

- 仅使用大赛提供的有标注数据(10万)和无标注数据(100万)

### 预训练模型

- 使用了 huggingface 上提供的 `hfl/chinese-macbert-base` 模型。链接为： https://huggingface.co/hfl/chinese-macbert-base

### 开源代码

- 使用了开源Smart对抗训练部分代码。链接：https://github.com/archinetai/smart-pytorch
- 使用了开源ema部分代码。链接：https://github.com/fadel/pytorch_ema
- 使用了QQ浏览器2021AI算法大赛赛道第一名方案的部分代码。链接：https://github.com/zr2021/2021_QQ_AIAC_Tack1_1st
- 使用了ALBEF的官方部分模型代码。链接：https://github.com/salesforce/ALBEF

### 算法描述

- 单流模型：
  1. 对于视觉特征使用了一个线性层和一个Relu函数，然后过了使用Mac-bert初始化的embedding层得到vision-embedding
  2. 对于文本特征在与视觉特征融合之前使用Mac-bert的embedding得到text-embedding
  3. 对两个embedding做cat处理，直接传入bert-encoder
  4. 对encoder的输出做torch.mean处理，再过一个Linear层(768->200)
- 双流模型
  1. 修改ALBEF的vision-encoder为一个随机初始化的transformer block
  2. 修改ALBEF的分类头为一个Linear层(768->200)
- 预训练
  1. 单流模型采用MLM和ITM策略。
  2. 双流模型未做预训练
- 模型融合
  1. B榜最终结果使用了5折单流模型(添加了smart对抗训练)、5折双流模型、一个smart+ema+全量的双流模型、一个ema+全量的双流模型做加权融合

### 性能

B榜测试性能：0.688608

### 训练流程

- 由于操作失误，预训练了两次，首先对单流模型做5轮预训练(MLM+ITM，学习率5e-5)，在对上一轮预训练中的最后一轮产出的模型做4轮预训练(MLM+ITM, 学习率3e-5)，最终loss在0.6左右
- 训练5折单流模型
- 训练5折双流模型
- 训练smart+ema+全量的双流模型
- 训练ema+全量的双流模型

### 测试流程

- 对于五折单流模型取第四轮模型做预测
- 对于五折双流模型取第三轮模型做预测
- 对于smart+ema+全量的双流模型取第四轮做预测
- 对于ema+全量的双流模型取第三轮做预测
