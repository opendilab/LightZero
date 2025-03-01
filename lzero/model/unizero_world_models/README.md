# UniZero World Model 

## Position Encoding

This section provides an overview of the position encoding strategies used in the UniZero World Model. Two configurable options are supported based on the setting of `self.config.rotary_emb`:

- **nn.Embedding (Absolute Position Encoding)**  
- **ROPE (Relative Position Encoding)**


### 1. nn.Embedding (Absolute Position Encoding)

When `self.config.rotary_emb` is set to **False**, the model uses `nn.Embedding` for position encoding. This method includes:

- **Embedding Initialization:**  
  A positional embedding layer is initialized with `nn.Embedding`, which maps each position index to a fixed-size embedding vector.

- **Context Length Limitation:**  
  Due to the context length constraints, the model retains only the most recent steps in the key-value cache (kv_cache).

- **Position Embedding Correction**  
  When reusing `kv_cache`, the position embeddings need to be reset to start from zero. The embeddings are adjusted using `pos_emb_diff_k` and `pos_emb_diff_v` to simulate the effect of relative position encoding. For example:

  - Suppose the inference length calculation is `5 * 2 = 10`, then the current position encoding in `kv_cache` would be:
    ```
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    ```
  - When adding new data requires removing the first 2 steps from `kv_cache`, the remaining position encoding would be:
    ```
    2, 3, 4, 5, 6, 7, 8, 9
    ```
  - Directly using the original data at this point would lead to duplicated or incorrect position encoding, such as:
    ```
    2, 3, 4, 5, 6, 7, 8, 9, 8, 9
    ```
  - To solve this issue, the implementation corrects the position encoding in `kv_cache` by resetting it to:
    ```
    0, 1, 2, 3, 4, 5, 6, 7
    ```

### 2. ROPE (Relative Position Encoding)

When `self.config.rotary_emb` is set to **True**, the model adopts ROPE for position encoding. Key aspects include:

- **ROPE Initialization:**  
  Precomputed frequency components are used to apply rotary positional embeddings to the query and key tensors.

- **Indexing by Episode Timestep:**  
  Positions are indexed by the episode’s timestep where both the state (`s`) and action (`a`) count as two steps.

- **ROPE Principle:**  
  The mechanism is based on the paper [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864).

#### Performance and Application

- **nn.Embedding:**  
  Performs better in environments with short-term dependencies (e.g., Pong, DMC Cartpole-Swingup).

- **ROPE:**  
  Provides greater flexibility and scalability, making it suitable for environments with long-term dependencies.

---

### Conclusion

The UniZero World Model provides configurable options for position encoding, allowing users to choose between absolute (nn.Embedding) and relative (ROPE) approaches based on their environment’s needs. The correction mechanism in the nn.Embedding approach ensures consistency in the kv_cache when new data is appended, while ROPE offers enhanced flexibility for longer dependency settings.