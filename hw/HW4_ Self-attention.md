# HW4: Self-attention

‍

# 模型架构

- input:

  - ​` mels: (batch size, length, 40)`
- ​`out = self.prenet(mels)` 

  - ​`self.prenet = nn.Linear(40, d_model)`

    - 对最后一个维度做线性变化
    - ​`(batch size, length, 40)`​ --->  `(batch size, length, d_model)`
- ​`out = out.permute(1, 0, 2)`

  - 改变维度
  - ​`(batch size, length, d_model)`​ ---> `(length, batch_size, d_model)`​ 为了符合`encoder_layer`的输入
- ​`out = self.encoder_layer(out)`

  - ​`self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, dim_feedforward=256, nhead=2)`
  - 一层attention计算后接入一个FNN(d_model -> 256 -> d_model) ，两个都有Residual connection与LayerNorm
- ​`out = out.transpose(0, 1)`

  - 交换第0维与第1维：换回来
  - ​`(length, batch_size, d_model)`​---> `(batch size, length, d_model)`
- ​`stats = out.mean(dim=1)`

  - 将输入的sequence(vector set)求平均聚合成一个向量
  -  `(batch size, length, d_model)`​ ->  `(batch size, d_model)`
- ​`out = self.pred_layer(stats)`

  ```python
    		self.pred_layer = nn.Sequential(
    			nn.Linear(d_model, d_model),
    			nn.ReLU(),
    			nn.Linear(d_model, n_spks),
    		)
  ```

  - 最后的做分类

# 提交记录

- v0

  ![image](assets/image-20260103192925-yhknmpg.png)

- v1：修改模型结构

  先增加encoder的层数以及multihead的个数增加模型的表达力

  encoder的层数: 1 -> 4

  multihead:2 -> 4

  ![image](assets/image-20260103212403-osem1h2.png)
- v2:v1的效果好，那就继续增加训练次数

  - "total_steps": 70000 -> 100000,

  ![image](assets/image-20260103222039-u3lmq8m.png)

‍
