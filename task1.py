#!/usr/bin/env python3

# This script trains a neural network that solves the following task:
# Given an input sequence XY[0-5]+ where X and Y are two given digits,
# the task is to count the number of occurrences of X and Y in the remaining
# substring and then calculate the difference #X - #Y.
#
# Example:
# Input: 1213211
# Output: 2 (3 - 1)
#
# This task is solved with a multi-head attention network.

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

SEQ_LEN = 5
VOCAB_SIZE = 6
NUM_TRAINING_STEPS = 25000
BATCH_SIZE = 64

# This function generates data samples as described at the beginning of the
# script
def get_data_sample(batch_size=1):
    random_seq = torch.randint(low=0, high=VOCAB_SIZE - 1,
                               size=[batch_size, SEQ_LEN + 2])
    
    ############################################################################
    # TODO: Calculate the ground truth output for the random sequence and store
    # it in 'gts'.
    ############################################################################
    gts = []
    for rs in random_seq:
        count_x = torch.count_nonzero(torch.where(rs[2:] == rs[0],1,0))
        count_y = torch.count_nonzero(torch.where(rs[2:] == rs[1],1,0))
        
        ground_truth = count_x - count_y
        
        if 0 > ground_truth:
            ground_truth += SEQ_LEN
            
        gts.append(ground_truth)
        
    gts = torch.as_tensor(gts)
    # Ensure that GT is non-negative
    ############################################################################
    # TODO: Why is this needed?
    ############################################################################

    return random_seq, gts

# Network definition
class Net(nn.Module):
    def __init__(self, num_encoding_layers=1, num_hidden=64, num_heads=4):
        super().__init__()

        self.embedding = nn.Embedding(VOCAB_SIZE, num_hidden)
        positional_encoding = torch.empty([SEQ_LEN + 2, 1])
        nn.init.normal_(positional_encoding)
        self.positional_encoding = nn.Parameter(positional_encoding,
                                                requires_grad=True)
        q = torch.empty([1, num_hidden])
        nn.init.normal_(q)
        self.q = nn.Parameter(q, requires_grad=True)
        self.encoding_layers = [ EncodingLayer(num_hidden, num_heads)
                                for _ in range(num_encoding_layers) ]
        self.decoding_layer = MultiHeadAttention(num_hidden, num_heads)
        self.c1 = nn.Conv1d(num_hidden + 1, num_hidden, 1)
        self.fc1 = nn.Linear(num_hidden, 2 * SEQ_LEN + 1)

    def forward(self, x):
        x = self.embedding(x)
        B = x.shape[0]
        ########################################################################
        # TODO: In the following lines we add a (trainable) positional encoding
        # to our representation. Why is this needed?
        # Can you think of another similar task where the positional encoding
        # would not be necessary?
        ########################################################################
        positional_encoding = self.positional_encoding.unsqueeze(0)
        positional_encoding = positional_encoding.repeat([B, 1, 1])
        x = torch.cat([x, positional_encoding], axis=-1)
        x = x.transpose(1, 2)
        x = self.c1(x)
        x = x.transpose(1, 2)
        for encoding_layer in self.encoding_layers:
            x = encoding_layer(x)
        q = self.q.unsqueeze(0).repeat([B, 1, 1])
        x = self.decoding_layer(q, x, x)
        x = x.squeeze(1)
        x = self.fc1(x)
        return x

class EncodingLayer(nn.Module):
    def __init__(self, num_hidden, num_heads):
        super().__init__()

        self.att = MultiHeadAttention(embed_dim=num_hidden, num_heads=num_heads)
        self.c1 = nn.Conv1d(num_hidden, 2 * num_hidden, 1)
        self.c2 = nn.Conv1d(2 * num_hidden, num_hidden, 1)
        self.norm1 = nn.LayerNorm([num_hidden])
        self.norm2 = nn.LayerNorm([num_hidden])

    def forward(self, x):
        x = self.att(x, x, x)
        x = self.norm1(x)
        x1 = x.transpose(1, 2)
        x1 = self.c1(x1)
        x1 = F.relu(x1)
        x1 = self.c2(x1)
        x1 = F.relu(x1)
        x1 = x1.transpose(1, 2)
        x += x1
        x = self.norm2(x)
        return x

# The following two classes implement Attention and Multi-Head Attention from
# the paper "Attention Is All You Need" by Ashish Vaswani et al.

################################################################################
# TODO: Summarize the idea of attention in a few sentences. What are Q, K and V?
#
# In the following lines, you will implement a naive version of Multi-Head
# Attention. Please do not derive from the given structure. If you have ideas
# about how to optimize the implementation you can however note them in a
# comment or provide an additional implementation.
################################################################################

class Attention(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()

        ########################################################################
        # TODO: Add necessary member variables.
        ########################################################################
        self.input_dim = input_dim
        self.embed_dim = embed_dim
    def forward(self, q, k, v):
        # q, k, and v are batch-first
        
        ########################################################################
        # TODO: First, calculate a trainable linear projection of q, k and v.
        # Then calculate the scaled dot-product attention as described in
        # Section 3.2.1 of the paper.
        ########################################################################
        k_transpose = torch.transpose(k,1,2)
        numerator = torch.matmul(q, k_transpose)
        attention_val = numerator/(self.embed_dim**0.5)
        softmax = torch.softmax(attention_val, dim=1)
        result = torch.matmul(softmax,v)
        
        return result

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        ########################################################################
        # TODO: Add necessary member variables.
        ########################################################################
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attention = Attention(None, embed_dim)
        
        self.query_attention = [nn.Linear(self.embed_dim, self.embed_dim, bias=False) for _ in range(num_heads)]
        self.key_attention = [nn.Linear(self.embed_dim, self.embed_dim,bias=False) for _ in range(num_heads)]
        self.value_attention = [nn.Linear(self.embed_dim, self.embed_dim, bias =False) for _ in range(num_heads)]
        self.output = nn.Linear(self.embed_dim*self.num_heads, self.embed_dim)
        

    def forward(self, q, k, v):
        # q, k, and v are batch-first

        ########################################################################
        # TODO: Implement multi-head attention as described in Section 3.2.2
        # of the paper.
        ########################################################################
        Query_attention_mat = [self.query_attention[_](q) for _ in range(self.num_heads)]
        Key_attention_mat = [self.key_attention[_](k) for _ in range(self.num_heads)]
        Value_attention_mat = [self.value_attention[_](v) for _ in range(self.num_heads)]
        
        Query_attention =[self.attention(Query_attention_mat[_], Key_attention_mat[_], Value_attention_mat[_]) for _ in range(self.num_heads)]
        concat = torch.cat(Query_attention, dim = -1)
        
        result = self.output(concat)
        return result

# Instantiate network, loss function and optimizer
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.005, momentum=0.9)

# Train the network
for i in range(NUM_TRAINING_STEPS):
    inputs, labels = get_data_sample(BATCH_SIZE)

    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    accuracy = (torch.argmax(outputs, axis=-1) == labels).float().mean()

    if i % 100 == 0:
        print('[%d/%d] loss: %.3f, accuracy: %.3f' %
              (i , NUM_TRAINING_STEPS - 1, loss.item(), accuracy.item()))
    if i == NUM_TRAINING_STEPS - 1:
        print('Final accuracy: %.3f, expected %.3f' %
              (accuracy.item(), 1.0))
