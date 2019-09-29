import math
import copy
import numpy as np

import torch
import torch.nn as nn

hidden_size = 256
max_position_embeddings = 16
num_hidden_layers = 2
intermediate_size = 256*4
num_attention_heads = 4
attention_probs_dropout_prob = 0.1
hidden_dropout_prob = 0.2
num_classes = 56


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class PositionalEncoding(nn.Module):
    def __init__(self):
        super(PositionalEncoding, self).__init__()
        emb_dim = hidden_size
        max_len = max_position_embeddings
        self.position_enc = self.position_encoding_init(max_len, emb_dim)

    @staticmethod
    def position_encoding_init(n_position, emb_dim):
        ''' Init the sinusoid position encoding table '''

        # keep dim 0 for padding token position encoding zero vector
        position_enc = np.array([
            [pos / np.power(10000, 2 * (j // 2) / emb_dim) for j in range(emb_dim)]
            if pos != 0 else np.zeros(emb_dim) for pos in range(n_position)])
        
        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # apply sin on 0th,2nd,4th...emb_dim
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # apply cos on 1st,3rd,5th...emb_dim
        return torch.from_numpy(position_enc).type(torch.FloatTensor)

    def forward(self, word_seq):
        position_encoding = self.position_enc.unsqueeze(0).expand_as(word_seq)
        position_encoding = position_encoding.to(word_seq.device)
        word_pos_encoded = word_seq + position_encoding
        return word_pos_encoded 


class BertSelfAttention(nn.Module):
    def __init__(self):
        super(BertSelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads #8
        self.attention_head_size = int(hidden_size / num_attention_heads) #64
        self.all_head_size = self.num_attention_heads * self.attention_head_size #512

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = BertLayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertAttention(nn.Module):
    def __init__(self):
        super(BertAttention, self).__init__()
        self.self_att = BertSelfAttention()
        self.output = BertSelfOutput()

    def forward(self, input_tensor, attention_mask):
        self_output = self.self_att(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor) # Residual layer
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = BertLayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertLayer(nn.Module):
    def __init__(self):
        super(BertLayer, self).__init__()
        self.attention = BertAttention()
        self.intermediate = BertIntermediate()
        self.output = BertOutput()

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class BertEncoder(nn.Module):
    def __init__(self):
        super(BertEncoder, self).__init__()
        layer = BertLayer()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None, output_all_encoded_layers=False):
        all_encoder_layers = []
        for layer_module in self.layers:
            hidden_states = layer_module(hidden_states, attention_mask)
            #if output_all_encoded_layers:
            #    all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

class BertPooler(nn.Module):
    def __init__(self):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class AttentionModule(nn.Module):
    def __init__(self):
        super(AttentionModule, self).__init__()                 
        self.embedding = PositionalEncoding()
        # Bert encoder
        self.encoder = BertEncoder()
        # Bert pooler
        self.pooler = BertPooler()

    def forward(self, x):
        x = self.embedding(x)
        encoded_layers = self.encoder(x)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        return pooled_output