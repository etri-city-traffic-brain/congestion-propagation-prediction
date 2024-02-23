import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import numpy as np

class AGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k):
        super(AGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights = nn.Parameter(torch.FloatTensor(2*cheb_k*dim_in, dim_out)) # 2 is the length of support / dim_out : rnn_unit or rnn_unit * 2 (64 or 128)
        self.bias = nn.Parameter(torch.FloatTensor(dim_out))
        nn.init.xavier_normal_(self.weights)
        nn.init.constant_(self.bias, val=0)
        
    def forward(self, x, supports):
        x_g = []
        support_set = []
        for support in supports:
            support_ks = [torch.eye(support.shape[0]).to(support.device), support]
            for k in range(2, self.cheb_k):
                support_ks.append(torch.matmul(2 * support, support_ks[-1]) - support_ks[-2]) 
            support_set.extend(support_ks)
        for support in support_set:
            x_g.append(torch.einsum("nm,bmc->bnc", support, x))
        x_g = torch.cat(x_g, dim=-1) # B, N, 2 * cheb_k * dim_in
        # print('x_g shape : ', x_g.shape)      # (64,207,585)
        # print('self.weights shape : ', self.weights.shape)
        x_gconv = torch.einsum('bni,io->bno', x_g, self.weights) + self.bias  # b, N, dim_out
        return x_gconv
    
class AGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out   # rnn_units : 64
        self.gate = AGCN(dim_in+self.hidden_dim, 2*dim_out, cheb_k)
        self.update = AGCN(dim_in+self.hidden_dim, dim_out, cheb_k)

    def forward(self, x, state, supports):
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, supports))   # paper r_t : z / paper u_t : r
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)    # z, r : (B, N, hidden)
        candidate = torch.cat((x, z*state), dim=-1)     # candidate : (B, N, hidden + input_dim)
        hc = torch.tanh(self.update(candidate, supports))   # paper C_t : hc
        h = r*state + (1-r)*hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)
    
class ADCRNN_Encoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, num_layers):
        super(ADCRNN_Encoder, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k))   # dim_out : rnn_unit (64)
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k))

    def forward(self, x, init_state, supports):
        #shape of x: (B, T, N, D), shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):    # num layer : 1
            state = init_state[i]   # init_state : (B,N,hidden)
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, supports)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        #output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #return current_inputs, torch.stack(output_hidden, dim=0)
        return current_inputs, output_hidden
    
    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return init_states

class ADCRNN_Decoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, num_layers):
        super(ADCRNN_Decoder, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Decoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k))

    def forward(self, xt, init_state, supports):
        # xt: (B, N, D)
        # init_state: (num_layers, B, N, hidden_dim)
        assert xt.shape[1] == self.node_num and xt.shape[2] == self.input_dim
        current_inputs = xt
        output_hidden = []
        for i in range(self.num_layers):
            state = self.dcrnn_cells[i](current_inputs, init_state[i], supports)
            output_hidden.append(state)
            current_inputs = state
        return current_inputs, output_hidden


class MegaCRN(nn.Module):
    # def __init__(self, num_nodes, input_dim, output_dim, horizon, rnn_units, original_matrix, num_layers=1, cheb_k=3,
    #              ycov_dim=1, mem_num=20, mem_dim=64, cl_decay_steps=2000, use_curriculum_learning=True):
    def __init__(self, num_nodes, input_dim, output_dim, horizon, rnn_units, num_layers=1, cheb_k=3,
                 ycov_dim=1, mem_num=20, mem_dim=64, cl_decay_steps=2000, use_curriculum_learning=True):
        super(MegaCRN, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim  # 1
        self.rnn_units = rnn_units
        self.output_dim = output_dim    # 1
        self.horizon = horizon
        self.num_layers = num_layers
        self.cheb_k = cheb_k
        self.ycov_dim = ycov_dim
        self.cl_decay_steps = cl_decay_steps
        self.use_curriculum_learning = use_curriculum_learning

        # self.original_matrix = torch.Tensor(original_matrix)    # 추가 코드

        # bahdanau attention 추가 코드 ================================================
        self.fc_d = nn.Linear(self.rnn_units * 2, self.rnn_units, bias=False)
        self.fc_e = nn.Linear(self.rnn_units, self.rnn_units, bias=False)
        # self.fc_c = nn.Linear(self.rnn_units, 32, bias=False)
        # self.weight_c = nn.Parameter(torch.FloatTensor(13248, 1), requires_grad=True)     # LA
        # self.weight_c = nn.Parameter(torch.FloatTensor(20800, 1), requires_grad=True)       # BAY
        # ===========================================================================

        # memory
        self.mem_num = mem_num
        self.mem_dim = mem_dim
        self.memory = self.construct_memory()

        # encoder
        self.encoder = ADCRNN_Encoder(self.num_nodes, self.input_dim, self.rnn_units, self.cheb_k, self.num_layers)
        
        # deocoder
        self.decoder_dim = self.rnn_units + self.mem_dim      # concatenate testing을 위한 주석 처리 (원래 코드)
        # self.decoder_dim = self.rnn_units                       # concatenate testing을 위한 dimension 변경 (test 코드)
        self.decoder = ADCRNN_Decoder(self.num_nodes, self.output_dim + self.ycov_dim, self.decoder_dim, self.cheb_k, self.num_layers)

        # output
        # self.proj = nn.Sequential(nn.Linear(self.decoder_dim, self.output_dim, bias=True))
        self.proj = nn.Sequential(
            nn.Linear(self.decoder_dim + 64, self.output_dim, bias=True)
        )

    def compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def construct_memory(self):
        memory_dict = nn.ParameterDict()
        memory_dict['Memory'] = nn.Parameter(torch.randn(self.mem_num, self.mem_dim), requires_grad=True)     # (M, d)
        memory_dict['Wq'] = nn.Parameter(torch.randn(self.rnn_units, self.mem_dim), requires_grad=True)    # project to query
        memory_dict['We1'] = nn.Parameter(torch.randn(self.num_nodes, self.mem_num), requires_grad=True)    # project memory to embedding
        memory_dict['We2'] = nn.Parameter(torch.randn(self.num_nodes, self.mem_num), requires_grad=True)    # project memory to embedding

        memory_dict['test_We1'] = nn.Parameter(torch.randn(self.mem_dim, self.rnn_units), requires_grad=True)    # project memory to embedding
        memory_dict['test_We2'] = nn.Parameter(torch.randn(self.mem_dim, self.rnn_units), requires_grad=True)    # project memory to embedding

        for param in memory_dict.values():
            nn.init.xavier_normal_(param)
        return memory_dict
    
    def query_memory(self, h_t:torch.Tensor):
        query = torch.matmul(h_t, self.memory['Wq'])     # (B, N, d)
        att_score = torch.softmax(torch.matmul(query, self.memory['Memory'].t()), dim=-1)         # alpha: (B, N, M)
        value = torch.matmul(att_score, self.memory['Memory'])     # (B, N, d)

        # _, ind = torch.topk(att_score, k=2, dim=-1)   # 원래 코드
        _, ind = torch.topk(att_score, k=20, dim=-1)    # 추가 코드

        pos = self.memory['Memory'][ind[:, :, 0]] # B, N, d

        # neg = self.memory['Memory'][ind[:, :, 1]] # B, N, d    원래 코드
        neg = self.memory['Memory'][ind[:, :, 1:]]      # 추가 코드

        return value, query, pos, neg, att_score
            
    def forward(self, x, y_cov, labels=None, batches_seen=None):
        node_embeddings1 = torch.matmul(self.memory['We1'], self.memory['Memory'])
        node_embeddings2 = torch.matmul(self.memory['We2'], self.memory['Memory'])
        g1 = F.softmax(F.relu(torch.mm(node_embeddings1, node_embeddings2.T)), dim=-1)
        g2 = F.softmax(F.relu(torch.mm(node_embeddings2, node_embeddings1.T)), dim=-1)

        supports = [g1, g2]       # 원래 코드 주석처리
        init_state = self.encoder.init_hidden(x.shape[0])
        h_en, state_en = self.encoder(x, init_state, supports) # B, T, N, hidden
        h_t = h_en[:, -1, :, :] # B, N, hidden (last state)

        h_att, query, pos, neg, attention_score = self.query_memory(h_t)

        # test 추가 코드 (M_t를 사용 하여 adjacency matrix 생성) ====================================
        # h_att shape : (B,N,d)
        node_embeddings3 = torch.matmul(h_att[-1,:,:], self.memory['test_We1'])
        node_embeddings4 = torch.matmul(h_att[-1,:,:], self.memory['test_We2'])

        test_g1 = F.softmax(F.relu(torch.mm(node_embeddings3, node_embeddings4.T)), dim=-1)
        test_g2 = F.softmax(F.relu(torch.mm(node_embeddings4, node_embeddings3.T)), dim=-1)
        # test_g1 = F.softmax(F.relu(torch.mm(h_att[-1,:,:], h_att[-1,:,:].T)), dim=-1)
        # test_g2 = F.softmax(F.relu(torch.mm(h_att[-1,:,:], h_att[-1,:,:].T)), dim=-1)

        # test_supports = [test_g1, test_g2, self.original_matrix]
        # test_g1 = test_g1 * 0.5 + self.original_matrix * 0.5
        # test_g2 = test_g2 * 0.5 + self.original_matrix * 0.5
        test_supports = [test_g1, test_g2]
        # ======================================================================================

        h_t = torch.cat([h_t, h_att], dim=-1)
        
        ht_list = [h_t]*self.num_layers
        go = torch.zeros((x.shape[0], self.num_nodes, self.output_dim), device=x.device)
        out = []
        for t in range(self.horizon):
            # h_de, ht_list = self.decoder(torch.cat([go, y_cov[:, t, ...]], dim=-1), ht_list, supports)  # 원래 코드
            h_de, ht_list = self.decoder(torch.cat([go, y_cov[:, t, ...]], dim=-1), ht_list, test_supports)

            # h_de shape : [64,207,128]
            # h_en shape : [64,12,207,64]

            # self.fc_d(h_de) shape : [64,207,64]
            # self.fc_e(h_en) shape : [64,12,207,64]
            # self.weight_c shape : [1,64]

            # attention_x shape : [64,12,13248] / [12,847872]
            # alignment_scores.shape : [64,12,1]
            # attn_weights shape : [64,12,1]
            # context_vector shape : [64,207,64]

            # ===============================================================================================
            # attention_x = torch.tanh(self.fc_d(h_de).unsqueeze(1) + self.fc_e(h_en)).reshape(64, 12, -1)
            # alignment_scores = attention_x.matmul(self.weight_c)
            # attn_weights = F.softmax(alignment_scores, dim=-1)
            # context_vector = torch.einsum('abcd, abe -> acd', h_en, attn_weights)
            #
            # attention_x = torch.tanh(self.fc_d(h_de).unsqueeze(1) + self.fc_e(h_en)).reshape(64, 12, -1)
            # alignment_scores = attention_x.matmul(self.weight_c).squeeze()
            # attn_weights = F.softmax(alignment_scores, dim=-1)
            # context_vector = torch.einsum('abcd, ab -> acd', h_en, attn_weights)
            # ================================================================================================

            # h_en shape : [64,12,207,64]
            # attention_x shape : [64,1,12]
            attention_x = torch.matmul(self.fc_d(h_de).unsqueeze(1).reshape(1, 1, -1), self.fc_e(h_en).reshape(1, 12, -1).transpose(1, 2))
            attention_weights = F.softmax(attention_x, dim=-1)
            context_vector = torch.einsum('abcd, aeb -> acd', h_en, attention_weights)

            # context_vector_trans = self.fc_c(context_vector)

            # h_de shape : [64,207,192]
            h_de = torch.cat([h_de, context_vector], dim=-1)

            go = self.proj(h_de)
            out.append(go)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self.compute_sampling_threshold(batches_seen):
                    go = labels[:, t, ...]
        output = torch.stack(out, dim=1)
        
        return output, h_att, query, pos, neg, attention_score

def print_params(model):
    # print trainable params
    param_count = 0
    print('Trainable parameter list:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape, param.numel())
            param_count += param.numel()
    print(f'In total: {param_count} trainable parameters. \n')
    return

def main():
    import sys
    import argparse
    from torchsummary import summary
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=3, help="which GPU to use")
    parser.add_argument('--num_variable', type=int, default=424, help='number of variables (e.g., 207 in METR-LA, 325 in PEMS-BAY)')
    parser.add_argument('--his_len', type=int, default=12, help='sequence length of historical observation')
    parser.add_argument('--seq_len', type=int, default=12, help='sequence length of prediction')
    parser.add_argument('--channelin', type=int, default=1, help='number of input channel')
    parser.add_argument('--channelout', type=int, default=1, help='number of output channel')
    parser.add_argument('--rnn_units', type=int, default=64, help='number of hidden units')
    args = parser.parse_args()
    device = torch.device("cuda:{}".format(args.gpu)) if torch.cuda.is_available() else torch.device("cpu")
    model = MegaCRN(num_nodes=args.num_variable, input_dim=args.channelin, output_dim=args.channelout, horizon=args.seq_len, rnn_units=args.rnn_units).to(device)
    summary(model, [(args.his_len, args.num_variable, args.channelin), (args.seq_len, args.num_variable, args.channelout)], device=device)
    print_params(model)
    
if __name__ == '__main__':
    main()
