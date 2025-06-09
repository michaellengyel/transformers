import math
import torch
import torch.nn as nn

from einops import rearrange


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int):
        super().__init__()

        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe.permute(1, 0, 2).requires_grad_(False)
        return x


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (Batch, Seq_len, d_model) --> (Batch, Seq_len, d_ff) -> (Batch, Seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h!"
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)  # Wq
        self.w_k = nn.Linear(d_model, d_model)  # Wk
        self.w_v = nn.Linear(d_model, d_model)  # Wv

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # (Batch, h, seq_len, d_k) -> (Batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)  # (Batch, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # (Batch, Seq_len, d_model) -> (Batch, Seq_len, h, d_k) -> (Batch, h, Seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # (Batch, h, Seq_len, d_k) -> (Batch, Seq_len, h, d_k) -> (Batch, Seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # (Batch, Seq_len, d_model) -> (Batch, Seq_len, d_model)
        return self.w_o(x)


class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = nn.LayerNorm(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class TransformerEncoder(nn.Module):
    def __init__(self, src_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048):
        super().__init__()

        # Create encoder blocks
        encoder_blocks = []
        for _ in range(N):
            encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
            feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
            encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
            encoder_blocks.append(encoder_block)
        self.encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))

        # Create the positional encoding layer
        self.src_pos = PositionalEncoding(d_model, src_seq_len)

    def forward(self, src):
        src = self.src_pos(src)
        src = self.encoder(src, None)
        return src


class SpatialTransformerEncoder(nn.Module):
    def __init__(self, src_seq_len, d_model, patch_size, image_shape):
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size
        self.image_shape = image_shape
        self.transformer_encoder = TransformerEncoder(src_seq_len, d_model)
        self.fc_in = nn.Linear(192, 512)
        self.fc_out = nn.Linear(512, 192)

    def patchify(self, image):
        """
        Args:
            image (torch.Tensor): Input image of shape (B, C, H, W), where B is the batch size, C is the number of channels,
                                  H is the image height, and W is the image width.
            patch_size (int): The size of each patch.

        Returns:
            patches (torch.Tensor): Patches extracted from the image with shape (B, num_patches, C, patch_size, patch_size),
                                   where num_patches is the number of patches in the image.
        """
        # Check for valid patch size
        if image.size(2) % self.patch_size != 0 or image.size(3) % self.patch_size != 0:
            raise ValueError("Image dimensions must be divisible by patch size.")

        # Reshape the image into patches
        patches = rearrange(image, 'b c (h s1) (w s2) -> b (h w) c s1 s2', s1=self.patch_size, s2=self.patch_size)

        return patches

    def depatchify(self, patches):
        """
        Args:
            patches (torch.Tensor): Patches with shape (B, num_patches, C, patch_size, patch_size), where
                                    B is the batch size, num_patches is the number of patches in the image,
                                    C is the number of channels, and patch_size is the patch size.
            image_shape (tuple): A tuple containing the target image shape (H, W) for reconstruction.
            patch_size (int): The size of each patch.

        Returns:
            image (torch.Tensor): Reconstructed image with shape (B, C, H, W), where B is the batch size,
                                C is the number of channels, H is the image height, and W is the image width.
        """
        # Calculate the number of patches in each dimension
        h, w = self.image_shape
        num_patches_h = h // self.patch_size
        num_patches_w = w // self.patch_size

        # Reshape patches back into the image
        image = rearrange(patches, 'b (ph pw) c patch_h patch_w -> b c (ph patch_h) (pw patch_w)',
                          ph=num_patches_h, pw=num_patches_w, patch_h=self.patch_size, patch_w=self.patch_size)

        return image

    def forward(self, x):
        x = self.patchify(x)
        input_shape = x.shape
        x = x.flatten(2)
        x = self.fc_in(x) * math.sqrt(self.d_model)
        x = self.transformer_encoder(x)
        x = self.fc_out(x)
        x = x.reshape(input_shape)
        x = self.depatchify(x)
        return x


class SpatialTransformerEncoderOfficial(nn.Module):
    def __init__(self, src_seq_len, d_model, patch_size, image_shape):
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size
        self.image_shape = image_shape
        self.pos_encoder = PositionalEncoding(d_model=d_model, seq_len=src_seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=512, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.fc_in = nn.Linear(d_model, d_model)
        #self.fc_out = nn.Linear(d_model, d_model)

    def patchify(self, image):
        """
        Args:
            image (torch.Tensor): Input image of shape (B, C, H, W), where B is the batch size, C is the number of channels,
                                  H is the image height, and W is the image width.
            patch_size (int): The size of each patch.

        Returns:
            patches (torch.Tensor): Patches extracted from the image with shape (B, num_patches, C, patch_size, patch_size),
                                   where num_patches is the number of patches in the image.
        """
        # Check for valid patch size
        if image.size(2) % self.patch_size != 0 or image.size(3) % self.patch_size != 0:
            raise ValueError("Image dimensions must be divisible by patch size.")

        # Reshape the image into patches
        patches = rearrange(image, 'b c (h s1) (w s2) -> b (h w) c s1 s2', s1=self.patch_size, s2=self.patch_size)

        return patches

    def depatchify(self, patches):
        """
        Args:
            patches (torch.Tensor): Patches with shape (B, num_patches, C, patch_size, patch_size), where
                                    B is the batch size, num_patches is the number of patches in the image,
                                    C is the number of channels, and patch_size is the patch size.
            image_shape (tuple): A tuple containing the target image shape (H, W) for reconstruction.
            patch_size (int): The size of each patch.

        Returns:
            image (torch.Tensor): Reconstructed image with shape (B, C, H, W), where B is the batch size,
                                C is the number of channels, H is the image height, and W is the image width.
        """
        # Calculate the number of patches in each dimension
        h, w = self.image_shape
        num_patches_h = h // self.patch_size
        num_patches_w = w // self.patch_size

        # Reshape patches back into the image
        image = rearrange(patches, 'b (ph pw) c patch_h patch_w -> b c (ph patch_h) (pw patch_w)',
                          ph=num_patches_h, pw=num_patches_w, patch_h=self.patch_size, patch_w=self.patch_size)

        return image

    def forward(self, x):
        x = self.patchify(x)
        input_shape = x.shape
        x = x.flatten(2)
        x = self.fc_in(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        # x = self.fc_out(x)
        x = x.reshape(input_shape)
        x = self.depatchify(x)
        x = torch.sigmoid(x)
        return x


class CnnEncode(nn.Module):
    def __init__(self):
        super(CnnEncode, self).__init__()
        # Convolutional layers to reduce spatial dimensions
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        return x


class CnnDecode(nn.Module):
    def __init__(self):
        super(CnnDecode, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=2)
        self.deconv2 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=3, padding=3)

    def forward(self, x):
        x = nn.functional.relu(self.deconv1(x))
        x = nn.functional.sigmoid(self.deconv2(x))
        return x


class SpatialTransformerEncoderConv(nn.Module):
    def __init__(self, src_seq_len, d_model, patch_size, image_shape):
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size
        self.image_shape = image_shape
        self.pos_encoder = PositionalEncoding(d_model=d_model, seq_len=src_seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=512, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=5)

        self.cnn_encode = CnnEncode()
        self.cnn_decode = CnnDecode()

    def forward(self, x):
        x = self.cnn_encode(x)  # (B, 64, 13, 13)
        input_shape = x.shape
        x = x.flatten(2).permute(0, 2, 1) * math.sqrt(self.d_model)  # (B, 64, 169)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.permute(0, 2, 1).reshape(input_shape)
        x = self.cnn_decode(x)
        return x


if __name__ == '__main__':

    model = SpatialTransformerEncoderConv(src_seq_len=169, d_model=64, patch_size=8, image_shape=(64, 64))

    x = torch.zeros(8, 3, 64, 64)
    yp = model(x)
    print(yp)
