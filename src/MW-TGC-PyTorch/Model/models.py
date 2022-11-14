from Model.layers import *
from torch.autograd import Variable

class GCLSTM_model(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_features, out_channels, adj,
                 output_dropout=0.5, GCN=False, bias=False):
        super(GCLSTM_model, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        self.encoder_lstm = Encoder_LSTM(input_dim, hidden_dim, out_features, out_channels, adj, GCN, bias=bias)

        decoder_dim = adj[0].size(0)
        self.decoder_lstm = Decoder_LSTM(output_size=decoder_dim, hidden_size=hidden_dim,
                                         output_dropout=output_dropout, bias=bias)

        self.adj = adj
        self.bias = bias

    def forward(self, x):
        # Initialize hidden state with zeros
        #######################
        #  USE GPU FOR MODEL  #
        #######################

        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(x.size(0), self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(x.size(0), self.hidden_dim))

        # initialize cell state
        if torch.cuda.is_available():
            c0 = Variable(torch.zeros(x.size(0), self.hidden_dim).cuda())
        else:
            c0 = Variable(torch.zeros(x.size(0), self.hidden_dim))

        outs = []
        cn = c0
        hn = h0
        for seq in range(x.size(1)):
            hn, cn = self.encoder_lstm(x[:, seq, :], hn, cn)
            outs.append(hn)
        decoder_hn = hn
        decoder_cn = cn

        decoder_outputs = []
        new_x = x[:, -1, :]
        for seq in range(x.size(1)):
            new_x, decoder_hn, decoder_cn = self.decoder_lstm(new_x, decoder_hn, decoder_cn)
            decoder_outputs.append(new_x)
        return decoder_outputs

