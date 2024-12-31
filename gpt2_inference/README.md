# GPT 2
A truncated version of Karapthy's work, this repository gets GPT-2 weights from hugging-face, loads them into a
minimalistic computational graph defined in pytorch, then runs autoregressive sampling based on some initialization
string

Source: https://github.com/karpathy/nanoGPT  
Video: https://www.youtube.com/watch?v=l8pRSuU81PU

---

# Notable Points

Tiktokenizer: https://tiktokenizer.vercel.app/?model=gpt2

* Tokenization done by tiktoken library
* <|endoftext|> as a special token
* Use of Flash Attention CUDA kernels: torch.nn.functional.scaled_dot_product_attention() (pytorch 2.0)
* By default, nn.Linear and nn.Embedding is initialized with: torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
* Use of nn.GELU() in feed-forward

---

# Visualization of Attention Masks

Visualization of the n-th head of the first 4 layers for a range of 256 using the <|endoftext|> as the initial token.  

![Teaser](./assets/attention.gif?raw=true)
