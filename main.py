import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from contextlib import nullcontext
import time
# hyperparameters
batch_size = 256 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------

#-----------混合精度训练----------------------------------------------
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

torch.backends.cuda.matmul.allow_tf32 = True #加上 TF32 白捡加速
torch.backends.cudnn.allow_tf32 = True

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
#-------------------------------------------------------------------

torch.manual_seed(1337)
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    
# here are all the unique characters that occur in this text
#---------排序并列出不重复的65个字母--------------
chars = sorted(list(set(text)))
vocab_size = len(chars)
#---------------------------------------------

#------------------创建字母与数字对应------------
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
#---------------------------------------------

#---encode("hello")--> [21, 5, 12, 12, 4]-----
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
#---------------------------------------------

#---------------train and test splits---------
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))# first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]
#---------------------------------------------

#------------------data loading---------------
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])   
    x, y = x.to(device), y.to(device)                         #挪到 GPU 的 VRAM
    return x, y
#---------------------------------------------

#-------测 loss，为后续评估做准备----------------
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval() #切到评估模式，影响 Dropout 和 BatchNorm
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
#---------------------------------------------- 

class Head(nn.Module):
    """ one head of self-attention """
    def __init__(self, head_size):
        super().__init__()
        #--------三个线形层把输入x变成Q，K，V三种表示-----------------
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        #--------------------------------------------------------
        
        #----------因果掩码（注册为buffer因为不需要训练）-------------（因为使用flash attn，以下不需要）
        #self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        #--------------------------------------------------------

        #--------------防止过拟合，随机丢弃一些注意力权重--------------
        #self.dropout = nn.Dropout(dropout)
        #--------------------------------------------------------
        self.dropout_p = dropout
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        #--------核心公式实现-------------------------------------- 
        #wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 #注意力缩放，防止点积数值过大，导致softmax进入梯度消失区
        #wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))# (B, T, hs) @ (B, hs, T) -> (B, T, T)
        #wei = F.softmax(wei, dim=-1) 
        #wei = self.dropout(wei)
        #v = self.value(x)
        #out = wei @ v
        #---------------------------------------------------------

        #-----------Flash attn-----------------------------------
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True        
            )
        #--------------------------------------------------------
        return out
        
#----------创建多个并行的attention head-----------------------------
class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
#-----------------------------------------------------------------
#对每个 token 独立做更深层的特征加工，不涉及和其他 token 的交流。
class FeedFoward(nn.Module):

    def __init__ (self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),#先放大，学完特征之后再缩小
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
        
    def forward(self, x):
        return self.net(x)
#------------------------------------------------------------------
        
#------------------Transformer Block---------------------------
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):

        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
#--------------------------------------------------------------

#--------模型实现-----------------------------------------------
class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)#建一张表，把字符索引变成向量
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head)for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)
#——————————————————————————————————————————————————————————————

#-----决定不同类型的层用什么方式初始化权重-----------------------------
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    #PyTorch 默认的初始化（Kaiming uniform）不一定是最优的。
    #GPT-2 论文里发现用固定的小标准差（0.02）初始化效果更好，训练更稳定。
#-----------------------------------------------------------------


    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)# (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))# (T,C)
        x = tok_emb + pos_emb# (B,T,C)
        x = self.blocks(x)# (B,T,C)
        x = self.ln_f(x)# (B,T,C)
        logits = self.lm_head(x)# (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)#因为cross entropy要求输入（N，C）的2维
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
    #idx is (B, T) array of indices
        for _ in range(max_new_tokens):

            idx_cond = idx[:, -block_size:]

            logits, loss = self(idx_cond)

            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)#从65个字母中抽取概率最高的那一个

            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

        return idx
            
model = GPTLanguageModel() #只触发init
m = model.to(device)       #遍历模型里所有的权重张量，把他们移到 GPU 显存

print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

#--------------训练循环---------------------------------------------------
start_time = time.time()
for iter in range(max_iters):

    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        train_ppl = math.exp(losses['train'])
        val_ppl = math.exp(losses['val'])
        print(f"step {iter}: "
              f"train loss {losses['train']:.4f} (ppl {train_ppl:.2f}), "
              f"val loss {losses['val']:.4f} (ppl {val_ppl:.2f})")

    xb, yb = get_batch('train')
    with ctx:
        logits, loss = model(xb, yb) #触发__call__
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

total_time = time.time() - start_time
print(f"\n训练总耗时: {total_time:.1f} 秒 ({total_time/60:.2f} 分钟)")
#--------------------------------------------------------------------------

#------------generate from the model-------------------------------
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
#------------------------------------------------------------------

    
            
            

        
        

        
          
