import torch

RANDOM_SEED = 12345
batch_size = 256
input_size = 63
embed_dim = 32
num_layers = 1
hidden_size = 32
workers = 2
learning_rate = 1e-2
epochs = 50
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
datatype = "mimic3"

