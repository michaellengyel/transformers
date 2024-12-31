from contextlib import nullcontext
import torch
import tiktoken
from model import GPT

start = "<|endoftext|>" # or "<|endoftext|>" or etc.
num_samples = 10  # number of samples to draw
max_new_tokens = 500  # number of tokens generated in each sample
temperature = 0.8  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200  # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'  # 'float32' or 'bfloat16' or 'float16'
compile = False  # use PyTorch 2.0 to compile the model to be faster

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# init from a given GPT-2 model
model = GPT.from_pretrained('gpt2', dict(dropout=0.0))
model.eval()
model.to(device)
if compile:
    model = torch.compile(model)  # requires PyTorch 2.0 (optional)

# let's assume gpt-2 encodings by default
print("No meta.pkl found, assuming GPT-2 encodings...")
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')
