# %% [markdown]
# Coding the self-attention mechanism

# %% [markdown]
# ![image.png](attachment:image.png)

# %% [markdown]
# ![image.png](attachment:image.png)

# %% [markdown]
# An RNN is a type of neural network where the output from previous steps is fed as input to the current step. This allows the network to maintain a form of memory, making it suitable for sequential data like text. However, RNNs can struggle with long-range dependencies and are often slow to train.

# %%
import torch

print(torch.__version__)

# %% [markdown]
# In an encoder–decoder RNN, the input text is fed into the encoder, which processes it sequentially. The encoder updates its hidden state (the internal values at the hidden layers) at each step, trying to capture the entire meaning of the input sentence in the final hidden state. The decoder then takes this final hidden state to start generating the translated sentence, one word at a time. It also updates its hidden state at each step, which is supposed to carry the context necessary for the next-word prediction.

# %% [markdown]
# RNNs work fine for translating short sentences, they don’t work well for lon-
# ger texts as they don’t have direct access to previous words in the input. One major
# shortcoming in this approach is that the RNN must remember the entire encoded
# input in a single hidden state before passing it to the decoder. 
# Hence, researchers developed the Bahdanau attention mechanism, which allows the decoder to focus on different parts of the input sequence at each step of the output generation. This is achieved by computing a set of attention weights that determine the importance of each input token for the current output token being generated.

# %% [markdown]
# A simple self-attention mechanism without trainable weights

# %% [markdown]
# ![image.png](attachment:image.png)

# %% [markdown]
# The goal of self-attention is to compute a context vector for each input element that combines information from all other input elements. In this example, we compute the context vector z(2). The importance or contribution of each input element for computing z(2) is determined by the attention weights 21 to 2T. When computing z(2), the attention weights are calculated with respect to input element x(2) and all other inputs.

# %% [markdown]
# We can think of the context vector as an enriched embedding vector that captures the relevant information of one element in relation to all other elements in the input sequence, allowing the model to make more informed predictions for each output token.

# %%
import torch

inputs = torch.tensor(
    [[0.43, 0.15, 0.89],
     [0.55, 0.87, 0.66],
     [0.57, 0.85, 0.64],
     [0.22, 0.58, 0.33],
     [0.77, 0.25, 0.10],
     [0.05, 0.80, 0.55]]
)

# %% [markdown]
# the first intermediate step is to compute the attention scores.
# ![image.png](attachment:image.png)

# %%
query = inputs[1]
attention_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attention_scores_2[i] = torch.dot(x_i, query)
print(attention_scores_2)

# %%
# next step is to normalize each of the attention scores. This is to obtain the attention weights which should sum to 1. this is useful for interpretation and maintaining training stability.

attention_weights_2_tmp = attention_scores_2 / attention_scores_2.sum()
print("Attention weights: ", attention_weights_2_tmp)
print("Sum: ", attention_weights_2_tmp.sum())

# %% [markdown]
# in practice, you would use a softmax function to compute the attention weights. This ensures that the weights are positive and sum to 1.

# %%
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attention_scores_2_naive = softmax_naive(attention_scores_2)
# print("Attention weights (naive softmax): ", attention_scores_2_naive)
# print("Sum: ", attention_scores_2_naive.sum())

# %% [markdown]
# this naive softmax implementation (softmax_naive) may encounter
# numerical instability problems, such as overflow and underflow, when dealing with
# large or small input values. Therefore, in practice, it’s advisable to use the PyTorch
# implementation of softmax, which has been extensively optimized for performance:

# %%
attention_weights_2 = torch.softmax(attention_scores_2, dim=0)
# print("Attention weights (PyTorch softmax): ", attention_weights_2)
# print("Sum: ", attention_weights_2.sum())

# %%
# now we can calculate the context vector by multiplying the embedded input tokens with the attention weights and then summing the resulting vectors

query = inputs[1]
context_vector_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vector_2 += attention_weights_2[i] * x_i
# print("Context vector: ", context_vector_2)

# %% [markdown]
# Computing attention weights for all input tokens

# %% [markdown]
# now, we have computed the context vector for the second input token. now we can compute the context vectors for all the input tokens.

# %%
attention_scores = torch.empty(6,6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attention_scores[i,j] = torch.dot(x_i, x_j)
# print("Attention scores: ", attention_scores)

# %%
# when computing the preceding attention score tensor, we used a for-loop. this is not the most efficient way to compute the attention scores. we can use matrix multiplication to compute the attention scores more efficiently.

attention_scores = inputs @ inputs.T
# print("Attention scores: ", attention_scores)


# %%
attention_weights = torch.softmax(attention_scores, dim=-1)
# print("Attention weights: ", attention_weights)

# %%
row_2_sum = attention_weights[1].sum()
# print("Row 2 sum: ", row_2_sum)
# print("Sum of all rows: ", attention_weights.sum(dim=-1))

# %%
all_context_vectors = attention_weights @ inputs
# print(all_context_vectors)

# %% [markdown]
# implementing self-attention with trainable weights

# %% [markdown]
# the most notable difference is the introduction of weight matrices that are updated during training. These weights are crucial so that the model can learn to produce the correct context vectors.

# %% [markdown]
# We  will  implement  the  self-attention  mechanism  step  by  step  by  introducing  the three trainable weight matrices Wq, Wk, and Wv. These three matrices are used to project the embedded input tokens, x
# (i), into query, key, and value vectors, respectively
# 
# ![Screenshot From 2025-10-31 05-44-44.png](<attachment:Screenshot From 2025-10-31 05-44-44.png>)

# %%
x_2 = inputs[1] #the second input element
d_in = inputs.shape[1] # the input embedding dimension d=3
d_out = 2 # the output embedding dimension d=2

# %%
torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
# we set the requires_grad attribute to False to reduce
# clutter in the outputs, but if we were to use the
# weight matrices in the training process, we would set it to True
# to update them

# %%
query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value
# print("Query:", query_2)

# %% [markdown]
# ### Weight parameters vs. attention weights 
# In the weight matrices W, the term “weight” is short for “weight parameters,” the values of a neural network that are optimized during training. This is not to be confused with the attention weights. As we already saw, attention weights determine the extent to which a context vector depends on the different parts of the input (i.e., to what extent the network focuses on different parts of the input). 
# In summary, weight parameters are the fundamental, learned coefficients that define
# the network’s connections, while attention weights are dynamic, context-specific values.
# 

# %%
keys = inputs @ W_key
values = inputs @ W_value
# print("Keys.shape:", keys.shape)
# print("Values.shape:", values.shape)

# %%
# now we compute the attention scores
keys_2 = keys[1]
attention_scores_22 = query_2.dot(keys_2)
# print("Attention scores:", attention_scores_22)

# %%
attention_scores_2 = query_2 @ keys.T
# print("Attention scores:", attention_scores_2)

# %% [markdown]
# as we can see, the second element in the output matches the attention score attention_scores_22 seen above.

# %% [markdown]
# now we want to go from the attention scores to the attention weights. we do this by scaling the attention scores and using the softmax function. However, now we scale the attention scores by dividing them by the square root of the dimensionality of the keys.

# %%
d_k = keys.shape[-1]
attention_weights_2 = torch.softmax(attention_scores_2 / d_k ** 0.5, dim=-1)
# print("Attention weights:", attention_weights_2)

# %% [markdown]
# final step is to compute the context vectors as a weighted sum over the value vectors

# %%
context_vector_2 = attention_weights_2 @ values
# print("Context vector:", context_vector_2)

# %% [markdown]
# Implementing the self-attention python class

# %%
import torch.nn as nn

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attention_scores = queries @ keys.T #omega
        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        context_vector = attention_weights @ values
        return context_vector

# %% [markdown]
# In this PyTorch code, SelfAttention_v1 is a class derived from nn.Module, which is a
# fundamental building block of PyTorch models that provides necessary functionalities
# for model layer creation and management. 
#  The __init__ method initializes trainable weight matrices (W_query, W_key, and
# W_value) for queries, keys, and values, each transforming the input dimension d_in to
# an output dimension d_out. 
#   During  the  forward  pass,  using  the  forward  method,  we  compute  the  attention
# scores (attention_scores) by multiplying queries and keys, normalizing these scores using
# softmax. Finally, we create a context vector by weighting the values with these normal-
# ized attention scores.

# %%
torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
# print(sa_v1(inputs))

# %% [markdown]
# as a quick check, we notice that the second row ([0.3061, 0.8210]) matches the contents of context_vector_2 in the last section

# %% [markdown]
# Self-Attention class using PyTorch's Linear layers~

# %%
import torch.nn as nn

class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attention_scores = queries @ keys.T #omega
        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        context_vector = attention_weights @ values
        return context_vector

# %%
torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
# print(sa_v2(inputs))

# %% [markdown]
# ### Hiding future words with causal attention
# Causal attention or masked attention is a form of self-attention that restricts a model to only consider previous and current inputs in a sequence when processing any given token when computing attention scores. This is different from the standard self-attention mechanism which allows the model to access the whole sequence at once..
# We will modify our standard s-a mechanism which is essential for LLMs

# %% [markdown]
# ![image.png](attachment:image.png)

# %%
# first step is to compute the attention weights using the softmax function

queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attention_scores = queries @ keys.T
attention_weights = torch.softmax(attention_scores / keys.shape[-1] ** 0.5, dim=-1)
# print("Attention weights:", attention_weights)

# %%
# next we use PyTorch's tril function to create a lower triangular matrix to mask out future tokens in the attention weights matrix

context_length = attention_scores.shape[0]
mask_simple = torch.tril(torch.ones((context_length, context_length)))
# print(mask_simple)

# %%
masked_simple = attention_weights * mask_simple
# print("Masked attention weights:", masked_simple)

# %%
# third step is to renormalize the masked attention weights so that they sum to 1
row_sums = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_normalized = masked_simple / row_sums
# print(masked_simple_normalized)

# %% [markdown]
# A more efficient way to implement the causal attention is to use a math property of the softmax function.
# ![image.png](attachment:image.png)

# %%
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attention_scores.masked_fill(mask.bool(), float('-inf'))
# print(masked)

# %%
# now we apply the softmax function to obtain the masked attention weights

attention_weights = torch.softmax(masked / keys.shape[-1] ** 0.5, dim=-1)
# print(attention_weights)

# %% [markdown]
# ### masking additional attention weights with dropout
# dropout in deep learning is a technique where randomly selected hidden layer units are ignored during training (dropping them out). this method helps to prevent overfitting by ensuring that a model does not become too reliant on any specific set of hidden units. dropout is ONLY used during training and is disabled after.

# %%
torch.manual_seed(123)
dropout = nn.Dropout(p=0.5)
example = torch.ones(6,6)
# print(dropout(example))

# %%
torch.manual_seed(123)
# print(dropout(attention_weights))

# %% [markdown]
# Implementing a compact causal attention class

# %%
batch = torch.stack((inputs, inputs), dim=0)  # shape (2, 6, 3)
# print("Batch shape:", batch.shape)

# %%
# this class is similar to the previous SelfAttention_v2 class but includes causal masking and dropout for regularization

class CausalSelfAttention(nn.Module):
    def __init__(self, d_int, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_int, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_int, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_int, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)

        # create the causal mask once during initialization
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)      # shape (b, num_tokens, d_out)
        queries = self.W_query(x) # shape (b, num_tokens, d_out)
        values = self.W_value(x)  # shape (b, num_tokens, d_out
        
        attention_scores = queries @ keys.transpose(1,2)
        attention_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], float('-inf'))
        attention_weights = torch.softmax(
            attention_scores / d_in ** 0.5, dim=-1
        )
        attention_weights = self.dropout(attention_weights)
        
        context_vector = attention_weights @ values
        return context_vector


# %%
torch.manual_seed(123)
context_length = batch.shape[1]
causal_sa = CausalSelfAttention(d_in, d_out, context_length, 0.0)
context_vectors = causal_sa(batch)

# print("context_vectors shape:", context_vectors.shape)

# %% [markdown]
# ![image.png](attachment:image.png)

# %% [markdown]
# ### Extending single-head attention to multi-head attention
# The term "multi-head" refers to dividing the attention mechanism into multiple "heads", each operating independently. A single causal attention module can be considered single-head attention where there is onlu one set of attention weights processing the input sequentially. We will build this by stacking multiple causal-attention modules. Then we will implement the same multi-head attention module in a more complicated but efficient way.

# %%
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalSelfAttention(
                d_in, d_out, context_length, dropout, qkv_bias
            ) for _ in range(num_heads)]
        )
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

# %%
torch.manual_seed(123)
context_length = batch.shape[1]
d_in, d_out = 3, 2
mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0,  num_heads=2)
context_vectors = mha(batch)

# print(context_vectors)
# print("context_vectors shape:", context_vectors.shape)

# %% [markdown]
# implementing multi-head attention with weight splits

# %%
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0, "d_out must be divisible by num_heads")
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_out, d_out)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)      # shape (b, num_tokens, d_out)
        queries = self.W_query(x) # shape (b, num_tokens, d_out)
        values = self.W_value(x)  # shape (b, num_tokens, d_out)

        # reshape for multi-head attention
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        
        keys = keys.transpose(1,2)      # shape (b, num_heads, num_tokens, head_dim)
        queries = queries.transpose(1,2) # shape (b, num_heads, num_tokens, head_dim)
        values = values.transpose(1,2)  # shape (b


        attention_scores = queries @ keys.transpose(2,3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        
        attention_scores.masked_fill_(mask_bool, -torch.inf)
        
        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        attention_weights = self.dropout(attention_weights)
        
        context_vector = (attention_weights @ values).transpose(1,2)
        
        context_vector = context_vector.contiguous().view(b, num_tokens, self.d_out)
        
        context_vector = self.out_proj(context_vector)
        return context_vector
    

# %%
torch.manual_seed(123)
batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
context_vectors = mha(batch)
# print(context_vectors)
# print("context_vectors shape:", context_vectors.shape)

# %% [markdown]
# Attention mechanisms transform input elements into enhanced context vector
# representations that incorporate information about all inputs.
# * A self-attention mechanism computes the context vector representation as a
# weighted sum over the inputs.
# In a simplified attention mechanism, the attention weights are computed via
# dot products.
# A dot product is a concise way of multiplying two vectors element-wise and then
# summing the products.
# Matrix multiplications, while not strictly required, help us implement computa-
# tions more efficiently and compactly by replacing nested for loops.
# In self-attention mechanisms used in LLMs, also called scaled-dot product
# attention, we include trainable weight matrices to compute intermediate trans-
# formations of the inputs: queries, values, and keys.
# When working with LLMs that read and generate text from left to right, we add
# a causal attention mask to prevent the LLM from accessing future tokens.
# In addition to causal attention masks to zero-out attention weights, we can add
# a dropout mask to reduce overfitting in LLMs.
# The attention modules in transformer-based LLMs involve multiple instances of
# causal attention, which is called multi-head attention.
# We can create a multi-head attention module by stacking multiple instances of
# causal attention modules.
# A more efficient way of creating multi-head attention modules involves batched
# matrix multiplications.

# %%



