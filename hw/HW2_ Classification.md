# HW2: Classification

# 代码

[hw2.ipynb](assets/hw2-20260101212641-iwg1t1e.ipynb)

# 调参过程

- sample code

  ![image](assets/image-20251231161120-uvlswaw.png)

- v1: 增加num_epoch: 10 -> 100；增加hidden_layers: 2 -> 10

  正确率没有什么变化，提前终止

  ![image](assets/image-20251231163916-ia82z52.png)![image](assets/image-20251231163933-txrpufm.png)

  ![image](assets/image-20251231164221-jcvpymj.png)
- v2: 增加hidden_dim: 64 -> 128；hidden_layers：10->16

  ![image](assets/image-20251231204827-mkmr6p2.png)

  ![image](assets/image-20251231204842-6lt4wpi.png)

- v3: 增加hidden_dim:128-512

  训练到30终止

  ![image](assets/image-20251231211528-eronav9.png)

- v4:增加BN

  模型太复杂，训练集表现好，但是测试集效果很差，**过拟合了**

  ![image](assets/image-20251231214531-m1mc24i.png)

  ![image](assets/image-20251231214723-gwvj0wq.png)
- v5:模型简化：hidden_dim:512-256，hidden_layer:16->8

  ![image](assets/image-20251231220025-tdn1iq2.png)

  还是**过拟合**

  ![image](assets/image-20251231220156-qbk6yfq.png)
- v6简化模型：hidden_layer:hidden_dim:256-128，hidden_layer:8->4

  ![image](assets/image-20251231221507-r5b26wx.png)

  不过拟合了，但是模型的能力也不行了...

  ![image](assets/image-20251231221640-xx61r9h.png)
- V7：hidden_layer:hidden_dim:128->512；concat_nframes：3 -> 11

  过拟合的问题

  ![image](assets/image-20251231223905-511mqxg.png)

  ![image](assets/image-20251231224013-jvnl4rf.png)

‍

- v8:V7：hidden_layer:hidden_dim:128->256；concat_nframes：11 -> 21

  还是过拟合

  ![image](assets/image-20251231230051-6lcvne5.png)

  ![image](assets/image-20251231230219-180u5e3.png)
- v9:防止过拟合

  增加：`nn.Dropout(0.3)`：30%的神经元的输出直接为0

  L2 正则（weight decay）`optimizer = torch.optim.Adam(     model.parameters(),     lr=learning_rate,     weight_decay=1e-4 )`

  ![image](assets/image-20260101145847-9gvhy88.png)

  ![image](assets/image-20260101145821-t2i485a.png)

  看上去还可以增加训练次数，并且loss降的有点慢，可以调高一点
- v10：LR = 1e-3, num_epochs = 100

  ![image](assets/image-20260101153600-n174ufu.png)

  感觉就是模型能力达到极限了，下次把丢弃的神经元比例调低一下

  ![image](assets/image-20260101153757-am5dq7t.png)
- v11：`nn.Dropout(0.3)`​ -> `nn.Dropout(0.2)`

  ![image](assets/image-20260101162011-xs0aslc.png)

  还是模型能力的问题，并且epoch设置太高了前面就能看出来这个问题

  ![image](assets/image-20260101163145-vngi8bm.png)

- v12：`nn.Dropout(0.2)`​ -> `nn.Dropout(0.1)` num_epoch 100->50

  ![image](assets/image-20260101165744-7in4ash.png)

  ![image](assets/image-20260101165806-cke5r0j.png)
- v13: hidden_dim:256->1024

  ![image](assets/image-20260101174034-a2lgr60.png)

- v14:hidden_dim:1024->2048

  ![image](assets/image-20260101174126-mk0wc87.png)

  ![image](assets/image-20260101174211-irn9qgt.png)
- v15:增加层数：hidden_layers:4->6

  ![image](assets/image-20260101175917-9jgg4gj.png)

  ![image](assets/image-20260101180006-vkmood3.png)

- v16:

  - lr:1e-3 -> 2e-4
  - 使用`CosineAnnealingWarmRestarts`

  ![image](assets/image-20260101193454-qo7gtmp.png)

- v17

  - ​`Relu`​ -> `LeakyReLU`

  ![image](assets/image-20260101195503-hwb5qlm.png)

  效果大提升，但是有过拟合问题

  ![image](assets/image-20260101195629-0i52uwc.png)
- v18

  - ​`nn.Dropout(0.2)`​ -> `nn.Dropout(0.3)`

  ![image](assets/image-20260101203340-39lj877.png)

- v19

  - 训练集占比:0.75->0.8

  - 模型层数:4 hidden_dim=1024

  ![image](assets/image-20260101203840-v8a8gkw.png)

  ![image](assets/image-20260101203854-79bvrxc.png)
- v20

  - num_epochs: 20 -> 100

  ![image](assets/image-20260101212529-plnng3v.png)

‍
