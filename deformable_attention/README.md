# Deformable attention

---

## DETR (DEtection TRansformer)

This paper is the precursor to Deformable-DETR

Paper: https://arxiv.org/abs/2005.12872  
GitHub:  https://github.com/facebookresearch/detr (see for pretrained models)  
Video: https://www.youtube.com/watch?v=T35ba_VXkMY  
Institute: Facebook AI Research (FAIR)  
Authors: Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko  
Date: 2020  

<p align="center">
  <img src="assets/detr.png" width="100%" />
</p>

---

## Deformable DETR

DETR + Deformable Attention

Paper: https://arxiv.org/abs/2005.12872  
GitHub: https://github.com/fundamentalvision/Deformable-DETR  
Video: https://www.youtube.com/watch?v=3M9mS_3eiaw  
Video: 


The star of the show is the custom autograd function: MSDeformAttnFunction.apply(). This function is called by the
MSDeformAttn(nn.Module) module's forward(). The MSDeformAttn is used in place of nn.MultiheadAttention.

The recommended environment setup to compile this was outdated, this is what worked for me:   

CUDA Version: 12.2   
Driver Version: 535.183.01  
NVIDIA GeForce RTX 3090

```
conda create -n deformable_detr python=3.10 pip
conda activate deformable_detr
conda install pytorch=2.1.0 torchvision=0.16.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
pip install "numpy<2"
# Build Module:
cd ./models/ops
sh ./make.sh
```

<p align="center">
  <img src="assets/deformable_detr.png" width="100%" />
</p>

### MSDeformAttnFunction.apply()  

**value:**  
torch.Size([2, 10723, 8, 32])  
(batch_size, num_queries, num_heads, dim_per_head)  
10723 is the sum of all feature vectors on all scales: (10723, 256), the 256 is split up to 8 heads.  
10723=(76×106)+(38×53)+(19×27)+(10×14)  

**value_spatial_shapes:**  
torch.Size([4, 2])  
(num_levels, 2) e.g. [[76, 106], [38, 53], [19, 27], [10, 14]]  
Contains the spatial dimensions (height, width) for each level of the feature maps. Used to reconstruct multi-scale
feature maps from the flattened value tensor.  

**value_level_start_index:**  
torch.Size([4,])
(num_levels,) e.g, [0,  8056, 10070, 10583]  
Specifies the starting index of each level within the flattened value tensor. Enables mapping between value and the
corresponding spatial levels.   

**sampling_locations:**  
torch.Size([2, 10723, 8, 4, 4, 2])  
(batch_size, num_queries, num_heads, num_levels, num_points, 2)  
Specifies the sampling locations (in normalized [x, y] coordinates) for each query across all levels, heads, and points.
Learnable offsets are added to these locations during training to refine the regions of interest.  

**attention_weights:**  
torch.Size([2, 10723, 8, 4, 4])  
(batch_size, num_queries, num_heads, num_levels, num_points)  
Specifies the weights for each sampling location across all heads, levels, and points. These weights are multiplied
with the sampled values to compute the weighted sum.  

**im2col_step:**  
{int} 64  
Splits operations across smaller batch chunks to optimize memory usage and computational efficiency  

**output:**  
torch.Size([2, 10723, 256])  
256 = 8 * 32 = num_heads * dim_per_head  
(batch_size, num_queries, embed_dim)  

```
output = MSDeformAttnFunction.apply(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step)
```


<p align="center">
  <img src="assets/deformable_attention.png" width="100%" />
</p>

---

## DAT (Deformable Attention Transformer)

Paper: https://arxiv.org/abs/2201.00520  
GitHub: https://github.com/LeapLabTHU/DAT  

---

## DAT++

Paper: https://arxiv.org/abs/2309.01430  

---

