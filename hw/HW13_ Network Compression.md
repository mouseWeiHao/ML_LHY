# HW13: Network Compression

[hw13.ipynb](assets/hw13-20260116105651-8bpyr5z.ipynb)

- **模型总参数数 ≤ 60,000**
- 包括：

  - trainable
  - non-trainable
- 必须用 `torchsummary` 统计

# 提交记录

- v0：

  ![image](assets/image-20260115204502-ezsp8fq.png)

- v1:Knowledge_Distillation

  ```python
  def loss_fn_kd(student_logits, labels, teacher_logits, alpha=0.5, temperature=1.0):
      ce_loss = F.cross_entropy(student_logits, labels)

      # KL divergence loss with soft labels
      kd_loss = F.kl_div(
          F.log_softmax(student_logits / temperature, dim=1),
          F.softmax(teacher_logits / temperature, dim=1),
          reduction='batchmean'
      ) * (temperature ** 2)

      # Total loss
      loss = alpha * kd_loss + (1 - alpha) * ce_loss
      return loss
  ```

  ![image](assets/image-20260115215700-2h015y4.png)

- v2:调参

  ```python
  loss_fn_kd(student_logits, labels, teacher_logits, alpha=0.7, temperature=3.0)
  ```

  ![image](assets/image-20260115221625-6186vfr.png)

  ‍

- v3:增大epoch 10->500

  ![image](assets/image-20260116105554-fly1uke.png)
