import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CBOW(torch.nn.Module):
    def __init__(self, word_size, embedding_dim):
        super(CBOW, self).__init__()

        self.embeddings = nn.Embedding(word_size, embedding_dim,device=device)
        self.linear1 = nn.Linear(embedding_dim, 256, device=device)
        self.activation_function1 = nn.ReLU()

        self.linear2 = nn.Linear(256, word_size,device=device)
        self.activation_function2 = nn.LogSoftmax(dim=-1)

    def forward(self, inputs):
        embeds = sum(self.embeddings(inputs)).view(1, -1)
        out = self.linear1(embeds)
        out = self.activation_function1(out)
        out = self.linear2(out)
        out = self.activation_function2(out)
        return out

    def get_word_emdedding(self, word):
        pass
    
    def make_context_vector(context, word_to_ix):
        idxs = [word_to_ix[w] for w in context]
        return torch.tensor(idxs, dtype=torch.long, device=device)
    
    def get_word_vector(self,word,word_to_ix):
        word_idx = word_to_ix[word]
        word_tensor = torch.tensor([word_idx], dtype=torch.long, device=device)
        return self.embeddings(word_tensor).detach().cpu().numpy()
