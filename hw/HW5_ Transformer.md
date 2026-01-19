# HW5: Transformer

[hw5.ipynb](assets/hw5-20260105205516-iej0y3r.ipynb)

> [!NOTE]
> **HW5 要用台湾大学内部网站，无法提交，只能看在验证集的表现**

# sample源码理解

- config：主要参数

  - ​**​`max_tokens`​**

    - 控制每个 batch 的 token 总数
    - 影响显存占用与梯度稳定性
    - 显存允许时尽量调大
  - ​**​`accum_steps`​**

    - 梯度累积步数
    - 有效 batch size \= `max_tokens × accum_steps`
    - 显存不足时优先调这个
  - ​**​`lr_factor`​**

    - Noam 学习率调度器的缩放因子
    - 决定最大学习率大小
    - loss 震荡就调小，收敛慢就调大
  - ​**​`lr_warmup`​**

    - warmup 步数
    - 影响训练初期稳定性
    - batch 大时可适当增大
  - ​**​`clip_norm`​**

    - 梯度裁剪阈值
    - 防止梯度爆炸

  - ​**​`beam`​**

    - beam search 的 beam size
    - 只影响推理阶段
    - 常见范围 4–8，是最便宜的 BLEU 提升手段
  - ​**​`max_len_a`​**

    - 输出最大长度的比例系数
    - 输出偏短可调大
  - ​**​`max_len_b`​**

    - 输出最大长度的常数项
    - 防止过早截断

- 数据预处理

  ```markdown
  原始文本
  → 文本清洗
  → 子词分词（SentencePiece）
  → fairseq 二进制化（data-bin）
  → 训练期数据加载
  → batch 构造（padding + teacher forcing）
  → 进入模型 embedding
  ```

# 作业

- v0:RNN版本

  ​`BLEU = `​**​`15.74`​**​` 47.1/23.0/12.1/6.7 (BP = 0.914 ratio = 0.918 hyp_len = 103415 ref_len = 112709)`
- v1:改为transformer架构（参数配置与原文一致）、增加max_epoch=15->30、学习率的调整

  |项目|论文|
  | ----------------------| ------|
  |Encoder layers|6|
  |Decoder layers|6|
  |d\_model|512|
  |d\_ff|2048|
  |heads|8|
  |Activation|ReLU|
  |Decoder weight tying|是|

  ```python
  def get_rate(d_model, step_num, warmup_step):
      return (d_model ** -0.5) * min(
          step_num ** -0.5,
          step_num * warmup_step ** -1.5
      )

  encoder = TransformerEncoder(args, src_dict, encoder_embed_tokens)
  decoder = TransformerDecoder(args, tgt_dict, decoder_embed_tokens)

  arch_args = Namespace(
      encoder_embed_dim=512,
      encoder_ffn_embed_dim=2048,
      encoder_layers=6,
      decoder_embed_dim=512,
      decoder_ffn_embed_dim=2048,
      decoder_layers=6,
      share_decoder_input_output_embed=True,
      dropout=0.3,
  )

  # HINT: these patches on parameters for Transformer
  def add_transformer_args(args):
      args.encoder_attention_heads=8
      args.encoder_normalize_before=True

      args.decoder_attention_heads=8
      args.decoder_normalize_before=True

      args.activation_fn="relu"
      args.max_source_positions=1024
      args.max_target_positions=1024

      # patches on default parameters for Transformer (those not set above)
      from fairseq.models.transformer import base_architecture
      base_architecture(arch_args)

  add_transformer_args(arch_args)

  add_transformer_args(arch_args)
  ```

  2026-01-05 16:05:54 | INFO | hw5.seq2seq | BLEU = **26.78** 59.5/34.4/20.8/13.3 (BP = 0.977 ratio = 0.977 hyp_len = 110101 ref_len = 112709)

‍
