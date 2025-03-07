# UniZero 世界模型 

## 位置编码

本节概述了 UniZero 世界模型中使用的位置编码策略。根据 `self.config.rotary_emb` 的设置，支持以下两种可配置选项：

- **nn.Embedding（绝对位置编码）**  
- **ROPE（相对位置编码）**

### 1. nn.Embedding（绝对位置编码）

当 `self.config.rotary_emb` 设置为 **False** 时，模型采用 `nn.Embedding` 进行位置编码。该方法包括：

- **Embedding 初始化：**  
  利用 `nn.Embedding` 初始化一个位置嵌入层，将每个位置索引映射为固定尺寸的嵌入向量。

- **上下文长度限制：**  
  由于 context_length 的限制，模型只保留键值缓存（kv_cache）中最近的步数。

- **位置嵌入矫正**  
  当复用 kv_cache 时，需要将位置嵌入重置为从零开始。通过 `pos_emb_diff_k` 和 `pos_emb_diff_v` 对嵌入进行调整，从而模拟相对位置编码的效果。例如：

  - 假设推理长度计算为 `5 * 2 = 10`，则当前 kv_cache 的位置编码为：
    ```
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    ```
  - 当添加新的数据需要移除 kv_cache 中最前面的 2 步后，剩余位置编码为：
    ```
    2, 3, 4, 5, 6, 7, 8, 9
    ```
  - 此时直接使用原数据会导致位置编码重复或错误，比如获得：
    ```
    2, 3, 4, 5, 6, 7, 8, 9, 8, 9
    ```
  - 为解决该问题，实现中将对 kv_cache 中的位置编码进行矫正，将其重置为：
    ```
    0, 1, 2, 3, 4, 5, 6, 7
    ```

### 2. ROPE（相对位置编码）

当 `self.config.rotary_emb` 设置为 **True** 时，模型采用 ROPE 进行位置编码。主要内容包括：

- **ROPE 初始化：**  
  使用预计算出的频率成分对查询和键的张量施加 rotary 位置嵌入。

- **基于剧集时间步的索引：**  
  每个位置根据剧集（episode）的时间步进行索引，状态 (`s`) 和动作 (`a`) 各占一个步长，共计两个步长。

- **ROPE 原理：**  
  该方法的原理基于论文 [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)。

#### 性能与应用

- **nn.Embedding：**  
  在依赖关系较短的环境（如 Pong、DMC Cartpole-Swingup）中表现更好。

- **ROPE：**  
  提供更高的灵活性和扩展性，适合处理依赖关系更长的环境。



### 结论

UniZero 世界模型为位置编码提供了灵活的配置选项，允许根据不同环境的需求选择绝对位置编码（nn.Embedding）或相对位置编码（ROPE）。其中，nn.Embedding 的矫正机制确保在 kv_cache 重复使用时不会引入累积的索引错误，而 ROPE 则在长依赖场景中展现了更强的灵活性。
