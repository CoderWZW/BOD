import torch
import torch.nn as nn
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from util.loss_torch import bpr_loss,l2_reg_loss
import time

class NCF(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(NCF, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['NeuMF'])
        self.mlp_layer = int(args['-mlp_layer'])
        # mlp insize and outsize
        sizes = str(args['-sizes']).split(',')
        self.sizes = [int(num) for num in sizes]
        self.model = NCFEncoder(self.data, self.emb_size, self.mlp_layer, self.sizes)

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            start_time = time.time()
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                model.train()
                rec_user_emb, rec_item_emb = model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                batch_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb) + l2_reg_loss(self.reg, user_emb,pos_item_emb)
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 1000 == 0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())
            model.eval()
            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
            end_time = time.time()
            print("One epoch Running time: %f s" % (end_time - start_time))
            if epoch % 5 == 0:
                self.fast_evaluation(epoch)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        with torch.no_grad():
            u = self.data.get_user_id(u)
            score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
            return score.cpu().numpy()




class NCFEncoder(nn.Module):
    def __init__(self, data, emb_size, mlp_layer, sizes):
        super(NCFEncoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.mlp_layer = mlp_layer
        self.sizes = sizes
        self.in_out = []
        for i in range(len(self.sizes)-1):
            self.in_out.append((self.sizes[i], self.sizes[i+1]))

        self._fc_layers = torch.nn.ModuleList()
        for in_size, out_size in self.in_out:
            self._fc_layers.append(torch.nn.Linear(emb_size*in_size, emb_size*out_size))

        self.embedding_dict = self._init_model()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_mf_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_mf_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
            'user_mlp_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_mlp_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        return embedding_dict

    def forward(self):
        mlp_embeddings = torch.cat([self.embedding_dict['user_mlp_emb'], self.embedding_dict['item_mlp_emb']], 0)

        for idx, _ in enumerate(range(len(self._fc_layers))):
            mlp_embeddings = self._fc_layers[idx](mlp_embeddings)
            mlp_embeddings = torch.nn.ReLU()(mlp_embeddings)
        user_mlp_embeddings = mlp_embeddings[:self.data.user_num]
        item_mlp_embeddings = mlp_embeddings[self.data.user_num:]        

        user_embeddings = torch.cat([self.embedding_dict['user_mf_emb'], user_mlp_embeddings], 1)
        item_embeddings = torch.cat([self.embedding_dict['item_mf_emb'], item_mlp_embeddings], 1)

        return user_embeddings, item_embeddings



