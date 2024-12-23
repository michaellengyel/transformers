# Language Translation
Benchmarking a from scratch implementation against the official pytorch Transformer  
Credit: https://github.com/hkproj/pytorch-transformer

---

### Execution
Tensorboard:
```
tensorboard --logdir outputs
```

Train:
```
python train.py
```

Evaluation:
```
python evaluation.py
```

Attention Visualization:
```
python attention_visualization.py
```

---

### Results

---

### Images

<p align="center">
  <img src="assets/src_1_350_512.png" alt="Centered Image">
</p>
<p align="center">
  Batch element [1] of encoder src (after embedding, positional encoding and dropout). Size(b, 350, 512).
</p>

<p align="center">
  <img src="assets/src_mask_20_350.png" alt="Centered Image">
</p>
<p align="center">
  Encoder src_key_padding_mask. Size(b, 350). Padded sections are True (white).
</p>

<p align="center">
  <img src="assets/memory_1_350_512.png" alt="Centered Image">
</p>
<p align="center">
  Encoder output (memory). Size(b, 350, 512).
</p>

<p align="center">
  <img src="assets/tgt_1_350_512.png" alt="Centered Image">
</p>
<p align="center">
  Batch element [1] of decoder src (after embedding, positional encoding and dropout). Size(b, 350, 512).
</p>

<p align="center">
  <img src="assets/tgt_mask_20_350.png" alt="Centered Image">
</p>
<p align="center">
  Decoder tgt_key_padding_mask. Padded sections are True (white). Size(b, 350).
</p>

<p align="center">
  <img src="assets/src_mask_20_350.png" alt="Centered Image">
</p>
<p align="center">
  Decoder memory_key_padding_mask (same as src_key_padding_mask). Padded sections are True (white). Size(b, 350).
</p>

<p align="center">
  <img src="assets/tgt_mask_350_350.png" alt="Centered Image">
</p>
<p align="center">
  Decoder tgt_mask (causal mask). Padded sections are True (white). Size(350, 350).
</p>

---


