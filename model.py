import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.bn(self.embed(features))
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        captions = captions[:, :-1]
        #batch_size
        batch_size = features.size(0)
        #hidden_state and cell state
        hidden_state = torch.zeros((1, batch_size, self.hidden_size)).cuda()
        cell_state = torch.zeros((1, batch_size, self.hidden_size)).cuda()
        # create embedding
        embeds = self.word_embeddings(captions)
        embeds = torch.cat((features.unsqueeze(1), embeds), dim=1) 
        # embeddings new shape : (batch_size, captions length - 1, embed_size)
        lstm_out, _ = self.lstm(embeds, (hidden_state, cell_state))
        outputs = self.linear(lstm_out)
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        sampled_ids = []

        for i in range(max_len):                                    # maximum sampling length
            hiddens, states = self.lstm(inputs, states)        # (batch_size, 1, hidden_size), 
            outputs = self.linear(hiddens.squeeze(1))          # (batch_size, vocab_size)
            predicted = outputs.max(1)[1]
            if predicted.item() == 1:
                break
            sampled_ids.append(predicted)
            inputs = self.word_embeddings(predicted)
            inputs = inputs.unsqueeze(1)                       # (batch_size, 1, embed_size)
        
        return [pred.item() for pred in sampled_ids]
