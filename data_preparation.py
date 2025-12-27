# %% [markdown]
# We will:
# 1. Prepare text for LLM training.
# 2. Split text into word and sub-word tokens.
# 3. Byte-pair encoding (BPE) to tokenize the text.
# 4. Sampling training examples with a sliding window.
# 5. Convert tokens into vectors that feed into the LLM.

# %% [markdown]
# ![image.png](attachment:image.png)

# %% [markdown]
# Deep neural network models, including LLMs, cannot process raw text
# directly. Since text is categorical, it isn't compatible with the mathematical
# operations used to implement and train neural networks. Therefore, we need a
# way to represent words as continuous-valued vectors. 

# %% [markdown]
# The concept of converting data into a vector format is often referred to as embedding. Using a specific neural network layer or another pretrained neural network model, we can embed different data types, for example, video, audio, and text

# %% [markdown]
# ![image.png](attachment:image.png)

# %% [markdown]
# It's important to note that different data formats require distinct embedding models. For example, an embedding model designed for text would not be suitable for embedding audio or video data.

# %% [markdown]
# At its core, an embedding is a mapping from discrete objects, such as words, images, or even entire documents, to points in a continuous vector space -- the primary purpose of embeddings is to convert non-numeric data into a format that neural networks can process.
# While word embeddings are the most common form of text embedding, there are also embeddings for sentences, paragraphs, or whole documents. Sentence or paragraph embeddings are popular choices for retrieval-augmented generation. Retrieval-augmented generation combines generation (like producing text) with retrieval (like searching an external knowledge base) to pull relevant information when generating text, which is a technique that is beyond the scope of this book. Since our goal is to train GPT-like LLMs, which learn to generate text one word at a time, we focus on word embeddings.

# %% [markdown]
# ![image.png](attachment:image.png)

# %%
with open("./the-verdict.txt", "r", encoding="utf-8") as file:
    raw_text = file.read()
# print("Total characters in the text:", len(raw_text))
# print("First 100 characters of the text:", raw_text[:100])

# %% [markdown]
# Our goal is to tokenize this 20,479-character short story into individual words
# and special characters that we can then turn into embeddings for LLM
# training

# %%
# using sample text, we can user the re.split function to split the text into words
import re
text = "Hello, world! This is a test."
result = re.split(r'(\s)', text)
# print("Split text:", result)

# %% [markdown]
# this result is a list of individual words, whitespaces, and punctuation marks. We will refrain from making all text lowercase because capitalization helps LLMs distinguish between proper nouns and common nouns, understand sentence structure, and learn to generate text with proper
# capitalization.

# %%
# modifying the regular expression splits on whitespaces (\s) and commas and periods

result = re.split(r'(\s|[,.])', text)
# print(result)

# %% [markdown]
# A small remaining issue is that the list still includes whitespace characters.
# Optionally, we can remove these redundant characters safely as follows:

# %%
result = [item for item in result if item.strip()]
# print(result)

# %% [markdown]
# Let's modify it a bit further so that it can also handle other types of
# punctuation, such as question marks, quotation marks, and the double-dashes
# we have seen earlier in the first 100 characters of Edith Wharton's short story,
# along with additional special characters:
# 

# %%
text = "Hello, world! This is a test."
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
# print("Final split text:", result)

# %%
# now we apply it to the text from the file

preprocessed = re.split(r'([,.?_!"()\']|--|\s)', raw_text)
preprocessed_text = [item.strip() for item in preprocessed if item.strip()]
# print(len(preprocessed_text), "tokens in the preprocessed text.")

# %%
# print("Preprocessed text:", preprocessed_text[:100])  # Display first 100 tokens for brevity

# %% [markdown]
# Converting tokens into token IDs

# %% [markdown]
# to map the previously created tokens to their corresponding IDs, we need a vocabulary that contains all the unique tokens in the text. The vocabulary is a dictionary where each token is associated with a unique integer ID. This mapping allows us to convert the tokens into numerical representations that can be processed by the LLM.

# %% [markdown]
# ![image.png](attachment:image.png)

# %%
all_words = sorted(list(set(preprocessed_text)))
vocab_size = len(all_words)
# print("Vocabulary size:", vocab_size)

# %%
vocab = {token: integer for integer, token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    # print(item)
    if i > 50:
        break

# %%
# now we will create a simple tokenizer with an encode and decode method to convert text to token IDs and vice versa

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self, text):
            preprocessed_text = re.split(r'([,.?_!"()\']|--|\s)', text)
            preprocessed_text = [item.strip() for item in preprocessed_text if item.strip()]
            ids = [self.str_to_int[s] for s in preprocessed_text]
            return ids

    def decode(self, ids):
            text = " ".join([self.int_to_str[i] for i in ids])
            text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
            return text

# %%
tokenizer = SimpleTokenizerV1(vocab)

text = """It's the last he painted, you know," Mrs. Gisburn said."""
ids = tokenizer.encode(text)
# print("Encoded IDs:", ids)

# %%
# now we use the decode method to convert the IDs back to text
decoded_text = tokenizer.decode(ids)
# print("Decoded text:", decoded_text)

# %%
# let us try to tokenize a sample text that is not part of the original text
# text = "Hello, do you like tea? I do!"
# tokenizer.encode(text)

# %% [markdown]
# The problem is that the word "Hello" was not used in the The Verdict short
# story. Hence, it is not contained in the vocabulary. This highlights the need to
# consider large and diverse training sets to extend the vocabulary when
# working on LLMs.

# %% [markdown]
# Adding special context tokens

# %% [markdown]
# Here we will modify the previous tokenizer to include special context tokens like unknown words.

# %%
all_tokens = sorted(list(set(preprocessed_text)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token: integer for integer, token in enumerate(all_tokens)}
# print("length of vocabulary with special tokens:", len(vocab.items()))

# %%
for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)

# %% [markdown]
# now we adjust the previous tokenizer to include special context tokens like unknown words. This is important because it allows the model to handle words that were not present in the training data, which is a common scenario in real-world applications.

# %%
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}

    def encode(self, text):
        preprocessed_text = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed_text = [item.strip() for item in preprocessed_text if item.strip()]
        preprocessed_text = [
            item if item in self.str_to_int
            else "<|unk|>" for item in preprocessed_text
        ]

        ids = [self.str_to_int[s] for s in preprocessed_text]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text

# %%
text1 = "Hello, do you like tea? I do!"
text2 = "In the sunlight terraces of the palace."

text = " <|endoftext|> ".join([text1, text2])
# print("Combined text:", text)

# %%
tokenizer = SimpleTokenizerV2(vocab)
# print(tokenizer.encode(text))

# %%
# print("Decoded text:", tokenizer.decode(tokenizer.encode(text)))

# %% [markdown]
# Byte-Pair Encoding (BPE)

# %% [markdown]
# next we will implement a more sophisticated tokenization
# scheme based on a concept called byte pair encoding (BPE). The BPE tokenizer covered in this section was used to train LLMs such as GPT-2, GPT-3, and the original model used in ChatGPT.

# %%
# !pip install -qU tiktoken

# %%
from importlib.metadata import version
# print("tiktoken version:", version("tiktoken"))

# %%
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

# %%
text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of some"
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
# print(integers)

# %%
strings = tokenizer.decode(integers)
# print("Decoded text:", strings)

# %% [markdown]
# #### Data sampling with a sliding window
# ![image.png](attachment:image.png)

# %% [markdown]
# we will now implement a datloader that fetches input-target pairs from the training dataset using a sliding window approach. This method allows us to create overlapping sequences of tokens, which is essential for training LLMs effectively. The sliding window technique helps the model learn context and relationships between words in the text, improving its ability to generate coherent and contextually relevant responses.

# %%
with open("./the-verdict.txt", "r", encoding="utf-8") as file:
    raw_text = file.read()
enc_text = tokenizer.encode(raw_text)
# print("Encoded text length:", len(enc_text))

# %%
# using a sample text size of 50 tokens for demonstration

enc_sample = enc_text[:50]

# %%
context_size = 4

x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]
# print(f"x: {x}")
# print(f"y:      {y}")

# %%
for i in range(1, context_size+1):
    context = enc_sample[:i]
    target = enc_sample[i]
    # print(context, "----->", target)

# %%
for i in range(1, context_size+1):
    context = enc_sample[:i]
    target = enc_sample[i]

    # print(tokenizer.decode(context), "---->", tokenizer.decode([target]))

# %%
import torch
from torch.utils.data import Dataset, DataLoader

# creating a custom dataset class for GPT training
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

# %%
# the following will use the GPTDatasetV1 class to load the input in batches

def create_dataloaderv1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=False):

    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return dataloader

# %%
# testing the dataloader with a batch size of 1 for an llm with a context size of 4

with open("./the-verdict.txt", "r", encoding="utf-8") as file:
    raw_text = file.read()
dataloader = create_dataloaderv1(raw_text, batch_size=1, max_length=4, stride=1)

data_iter = iter(dataloader)
first_batch = next(data_iter)
# print("Input IDs:", first_batch[0])
# print("Target IDs:", first_batch[1])

# %%
second_batch = next(data_iter)
# print("Input IDs:", second_batch[0])
# print("Target IDs:", second_batch[1])

# %%
# looking at how to use the dataloader to sample with  a batch size greater than 1

dataloader = create_dataloaderv1(raw_text, batch_size=8, max_length=4, stride=4)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
# print("Input IDs:", inputs)
# print("Target IDs:", targets)

# %% [markdown]
# Creating token embeddings

# %% [markdown]
# the last step in preparing the data for training an LLM is to convert the tokens into embeddings. Token embeddings are continuous vector representations of tokens that capture their semantic meaning and relationships. These embeddings are essential for the LLM to understand and generate text effectively.

# %%
# suppose we have the following four input ids (after tokenization)
input_ids = ([2, 3, 5, 1])

# %%
vocab_size = 6
output_dim = 3

torch.manual_seed(42)  # For reproducibility
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# %%
# print(embedding_layer.weight)

# %%
vocab_size = 50257
output_dim = 256

token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# %%
max_length = 4
dataloader = create_dataloaderv1(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)

# %%
# print("Token ids:", inputs)
# print("\nInput shape:\n", inputs.shape)

# %%
token_embeddings = token_embedding_layer(inputs)
# print("Token embeddings shape:", token_embeddings.shape)

# %%
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

# print(pos_embedding_layer.weight)

# %%
pos_embeddings = pos_embedding_layer(torch.arange(max_length))
# print(pos_embeddings.shape)

# %%
input_embeddings = token_embeddings + pos_embeddings
# print(input_embeddings.shape)

# uncomment & execute the following line to see how the embeddings look like
# print(input_embeddings)

# %%



