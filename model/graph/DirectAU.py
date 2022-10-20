import torch
import torch.nn as nn
from base.graph_recommender import GraphRecommender
from util.sampler import next_batch_pairwise,next_batch_pointwise
from util.loss_torch import bpr_loss,l2_reg_loss,alignment_loss,uniformity_loss,InfoNCE
from base.torch_interface import TorchGraphInterface
import torch.nn.functional as F
import time

class DirectAU(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(DirectAU, self).__init__(conf, training_set, test_set)
        self.model = Matrix_Factorization(self.data, self.emb_size)
        #self.model = LGCN_Encoder(self.data, self.emb_size, 2)
        #self.model = SimGCL_Encoder(self.data, self.emb_size, 0.1, 2)

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)      
        for epoch in range(self.maxEpoch):
            start_time = time.time()
            for n, batch in enumerate(next_batch_pointwise(self.data, self.batch_size)):
                    u_idx, i_idx, y = batch
                    pos_u_idx = []
                    pos_i_idx = []
                    for index in range(len(y)):
                        if y[index] == 1:
                            pos_u_idx.append(u_idx[index])
                            pos_i_idx.append(i_idx[index])
                    #print(batch, pos_u_idx, pos_i_idx)
                    model.train()
                    rec_user_emb, rec_item_emb = model()
                    user_emb, item_emb = rec_user_emb[u_idx], rec_item_emb[i_idx]
                    pos_user_emb, pos_item_emb = rec_user_emb[pos_u_idx], rec_item_emb[pos_i_idx]
                    alignment = alignment_loss(pos_user_emb, pos_item_emb)
                    uniformity = self.reg * (uniformity_loss(user_emb) + uniformity_loss(item_emb)) / 2
                    #cl_loss = 0.001 * model.cal_cl_loss([u_idx,pos_u_idx])
                    l2 = l2_reg_loss(0.001, user_emb, pos_item_emb) 
                    # print("a", alignment)
                    # print("u", uniformity)
                    #batch_loss = alignment +  uniformity + cl_loss +l2
                    batch_loss = alignment +  uniformity +l2
                    optimizer.zero_grad()
                    batch_loss.backward()
                    optimizer.step()
                    if n % 1000 == 0:
                        print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())
            # for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
            #     user_idx, pos_idx, neg_idx = batch
            #     print(type(batch))
            #     model.train()
            #     rec_user_emb, rec_item_emb = model()
            #     user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
            #     batch_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb) + l2_reg_loss(self.reg, user_emb,pos_item_emb)
            #     # Backward and optimize
            #     optimizer.zero_grad()
            #     batch_loss.backward()
            #     optimizer.step()
            #     if n % 100 == 0:
            #         print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())
            model.eval()
            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
            end_time = time.time()
            print("Running time: %f s" % (end_time - start_time))
            if epoch % 1 == 0:
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


class Matrix_Factorization(nn.Module):
    def __init__(self, data, emb_size):
        super(Matrix_Factorization, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.embedding_dict = self._init_model()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        return embedding_dict

    def forward(self):
        return self.embedding_dict['user_emb'], self.embedding_dict['item_emb']



class LGCN_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers):
        super(LGCN_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        return embedding_dict

    def forward(self):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]
        for k in range(self.layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings = all_embeddings[:self.data.user_num]
        item_all_embeddings = all_embeddings[self.data.user_num:]
        return user_all_embeddings, item_all_embeddings


class SimGCL_Encoder(nn.Module):
    def __init__(self, data, emb_size, eps, n_layers):
        super(SimGCL_Encoder, self).__init__()
        self.data = data
        self.eps = eps
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })
        return embedding_dict

    def forward(self, perturbed=False):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = []
        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            if perturbed:
                random_noise = torch.rand_like(ego_embeddings).cuda()
                ego_embeddings += torch.sign(ego_embeddings) * F.normalize(random_noise, dim=-1) * self.eps
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.data.user_num, self.data.item_num])
        return user_all_embeddings, item_all_embeddings

    def cal_cl_loss(self, idx):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        user_view_1, item_view_1 = self.forward(perturbed=True)
        user_view_2, item_view_2 = self.forward(perturbed=True)
        user_cl_loss = InfoNCE(user_view_1[u_idx], user_view_2[u_idx], 0.2)
        item_cl_loss = InfoNCE(item_view_1[i_idx], item_view_2[i_idx], 0.2)
        return user_cl_loss + item_cl_loss


