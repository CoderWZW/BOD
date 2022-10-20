from util.conf import OptionConf
from tkinter.tix import Tree
import torch
import torch.nn as nn
from base.graph_recommender import GraphRecommender
from util.sampler import next_batch_pairwise, next_batch_pointwise,sample_batch_pointwise
from util.loss_torch import bpr_loss,l2_reg_loss,alignment_loss,uniformity_loss,alignment_loss_weight,alignment_loss_weight_1,InfoNCE
import torch.nn.functional as F
from base.torch_interface import TorchGraphInterface
import numpy as np
import time
from data.augmentor import GraphAugmentor
from data.ui_graph import Interaction




def match_loss(gw_syn, gw_real, dis_metric):
    dis = torch.tensor(0.0).to('cuda')

    if dis_metric == 'ours':

        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += distance_wb(gwr, gws)

    elif dis_metric == 'mse':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum((gw_syn_vec - gw_real_vec)**2)

    elif dis_metric == 'cos':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001)

    else:
        exit('DC error: unknown distance function')

    return dis

def distance_wb(gwr, gws):
    shape = gwr.shape

    # TODO: output node!!!!
    if len(gwr.shape) == 2:
        gwr = gwr.T
        gws = gws.T

    if len(shape) == 4: # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2: # linear, out*in
        tmp = 'do nothing'
    elif len(shape) == 1: # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        return 0

    dis_weight = torch.sum(1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
    dis = dis_weight
    return dis
    
class QGrace(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(QGrace, self).__init__(conf, training_set, test_set)
        self.trainmodel = self.config['trainmodel']
        self.generator = self.config['generator']
        self.datasetname = self.config['dataset.name']
        args = OptionConf(self.config['GM_AU'])
        self.generator_lr = float(args['-generator_lr'])
        self.generator_reg = float(args['-generator_reg'])
        self.generator_emb_size = int(args['-generator_emb_size'])
        #self.generator_save = int(args['-generator_save'])
        self.outer_loop = int(args['-outer_loop'])
        self.inner_loop = int(args['-inner_loop'])
        if self.trainmodel == "MF":
            self.model = Matrix_Factorization(self.data, self.emb_size)
        elif self.trainmodel == "LightGCN":
            self.model = LGCN_Encoder(self.data, self.emb_size,2)
        elif self.trainmodel == "SimGCL":
            self.model = SimGCL_Encoder(self.data, self.emb_size, 0.1, 2)
        elif self.trainmodel == 'SGL':
            self.model = SGL_Encoder(self.data, self.emb_size, 0.1, 2, 0.2, 2)
        else:
            print('Wrong trainmodel parameters!')
            exit(-1)

        if self.generator == "MF":
            self.model_full = GraphGenerator_MF(self.data, self.generator_emb_size)
        elif self.generator == "MLP":
            self.model_full = GraphGenerator_MLP(self.data, self.generator_emb_size)
        elif self.generator == "VAE":
            self.model_full = GraphGenerator_VAE(self.data, self.generator_emb_size)
        else:
            print('Wrong generator parameters!')
            exit(-1)



    def train(self):
        model = self.model.cuda()
        rec_user_emb, rec_item_emb = model()
        model_parameters = list(model.parameters())
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        
        if self.generator == "MF":
            model_full = self.model_full.cuda()
            rec_user_emb_full, rec_item_emb_full = model_full()
        elif self.generator == "MLP":
            model_full = self.model_full.cuda()
        elif self.generator == "VAE":
            model_full = self.model_full.cuda()
        else:
            print('Wrong generator parameters!')
            exit(-1)

        optimizer_full = torch.optim.Adam(model_full.parameters(), lr=self.generator_lr)
        ol_batch_size = self.batch_size


        for ol in range(self.maxEpoch):
            print("start generator training...")
            #start_time = time.time()
            for ol_iter in range(self.outer_loop):
                loss = torch.tensor(0.0).to('cuda')
                model_full.train()
                batch_ol = sample_batch_pointwise(self.data, ol_batch_size)
                u_idx_ol, i_idx_ol, y_ol = batch_ol
                pos_u_idx_ol = []
                pos_i_idx_ol = []
                for index in range(len(y_ol)):
                    if y_ol[index] == 1:
                        pos_u_idx_ol.append(u_idx_ol[index])
                        pos_i_idx_ol.append(i_idx_ol[index])        

                user_emb_ol, item_emb_ol = rec_user_emb[u_idx_ol], rec_item_emb[i_idx_ol]
                pos_user_emb_ol, pos_item_emb_ol = rec_user_emb[pos_u_idx_ol], rec_item_emb[pos_i_idx_ol]

                alignment_ol = alignment_loss(pos_user_emb_ol, pos_item_emb_ol)
                gw_real = torch.autograd.grad(alignment_ol, model_parameters, retain_graph=True, create_graph=True)
                
                if self.generator == "MF":
                    user_emb_full_ol, item_emb_full_ol = rec_user_emb_full[u_idx_ol], rec_item_emb_full[i_idx_ol]
                    pos_user_emb_full_ol, pos_item_emb_full_ol = rec_user_emb_full[pos_u_idx_ol], rec_item_emb_full[pos_i_idx_ol]       
                    alignment_syn_ol = alignment_loss_weight(user_emb_ol, item_emb_ol, user_emb_full_ol, item_emb_full_ol)
                elif self.generator == "MLP" or self.generator == "VAE":
                    A_weight_user_item = model_full(user_emb_ol, item_emb_ol)
                    alignment_syn_ol = alignment_loss_weight_1(user_emb_ol, item_emb_ol, A_weight_user_item)
                else:
                    print('Wrong generator parameters!')
                    exit(-1)
                gw_syn = torch.autograd.grad(alignment_syn_ol, model_parameters, retain_graph=True, create_graph=True)
                loss = match_loss(gw_real, gw_syn, 'ours')

                loss_reg = l2_reg_loss(self.generator_reg, user_emb_ol, item_emb_ol)
                loss = loss + loss_reg

                optimizer_full.zero_grad()
                loss.backward(retain_graph=True)
                optimizer_full.step()
                print('epoch:', ol, 'ol:', ol_iter, 'loss_real:', alignment_ol.item(), 'loss_syn:', alignment_syn_ol.item())  
                print('Gradient matching loss:', loss.item())
            model_full.eval()
            
            if ol == self.maxEpoch - 1:            
                break
            
            print("start model training...")

            for j in range(self.inner_loop):
                if self.trainmodel == "SGL":
                    dropped_adj1 = model.graph_reconstruction()
                    dropped_adj2 = model.graph_reconstruction()
                for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                    u_idx, pos_i_idx, neg_i_idx = batch
                    pos_u_idx = u_idx
                    neg_u_idx = u_idx
                    
                    model.train()
                    pos_user_emb_syn, pos_item_emb_syn = rec_user_emb[pos_u_idx], rec_item_emb[pos_i_idx]
                    neg_user_emb_syn, neg_item_emb_syn = rec_user_emb[neg_u_idx], rec_item_emb[neg_i_idx]

                    if self.generator == "MF":
                        pos_user_emb_full, pos_item_emb_full = rec_user_emb_full[pos_u_idx], rec_item_emb_full[pos_i_idx]
                        alignment_inner = alignment_loss_weight(pos_user_emb_syn, pos_item_emb_syn, pos_user_emb_full, pos_item_emb_full)
                    elif self.generator == "MLP" or self.generator == "VAE":
                        #A_weight_user_item_full = model_full(pos_u_idx, pos_i_idx)
                        A_weight_user_item_full = model_full(pos_user_emb_syn, pos_item_emb_syn)
                        A_weight_inner_full_detach = A_weight_user_item_full.detach()
                        alignment_inner = alignment_loss_weight_1(pos_user_emb_syn, pos_item_emb_syn, A_weight_inner_full_detach)
                    else:
                        print('Wrong generator parameters!')
                        exit(-1)
                    uniformity_inner = (uniformity_loss(pos_user_emb_syn) + uniformity_loss(neg_user_emb_syn) + uniformity_loss(pos_item_emb_syn) + uniformity_loss(neg_item_emb_syn)) / 4
                    if self.trainmodel == "SimGCL":
                        cl_loss = 0.005 * model.cal_cl_loss([u_idx,pos_i_idx])
                    elif self.trainmodel == "SGL":
                        cl_loss = 0.005 * model.cal_cl_loss([u_idx,pos_i_idx],dropped_adj1,dropped_adj2)
                    else:
                        cl_loss = 0

                    batch_loss_inner = alignment_inner + self.reg * uniformity_inner + cl_loss
                    optimizer.zero_grad()
                    rec_user_emb, rec_item_emb = model.forward()
                    batch_loss_inner.backward()
                    optimizer.step()
                    if n % 1000 == 0:
                        print('epoch:', ol, 'inner_training:', j + 1, 'inner_batch：', n)
                        print("alignment_inner:", alignment_inner.item(), "uniformity_inner：", uniformity_inner.item())
                        print('inner_batch_loss:', batch_loss_inner.item())
                model.eval()
                
                with torch.no_grad():
                        self.user_emb, self.item_emb = self.model()
                #end_time = time.time()
                #print("One epoch Running time: %f s" % (end_time - start_time))
                if ol % 5 == 0:
                    self.fast_evaluation(j)
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

class SGL_Encoder(nn.Module):
    def __init__(self, data, emb_size, drop_rate, n_layers, temp, aug_type):
        super(SGL_Encoder, self).__init__()
        self.data = data
        self.drop_rate = drop_rate
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.temp = temp
        self.aug_type = aug_type
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

    def graph_reconstruction(self):
        if self.aug_type==0 or 1:
            dropped_adj = self.random_graph_augment()
        else:
            dropped_adj = []
            for k in range(self.n_layers):
                dropped_adj.append(self.random_graph_augment())
        return dropped_adj

    def random_graph_augment(self):
        dropped_mat = None
        if self.aug_type == 0:
            dropped_mat = GraphAugmentor.node_dropout(self.data.interaction_mat, self.drop_rate)
        elif self.aug_type == 1 or self.aug_type == 2:
            dropped_mat = GraphAugmentor.edge_dropout(self.data.interaction_mat, self.drop_rate)
        dropped_mat = self.data.convert_to_laplacian_mat(dropped_mat)
        return TorchGraphInterface.convert_sparse_mat_to_tensor(dropped_mat).cuda()

    def forward(self, perturbed_adj=None):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]
        for k in range(self.n_layers):
            if perturbed_adj is not None:
                if isinstance(perturbed_adj,list):
                    ego_embeddings = torch.sparse.mm(perturbed_adj[k], ego_embeddings)
                else:
                    ego_embeddings = torch.sparse.mm(perturbed_adj, ego_embeddings)
            else:
                ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.data.user_num, self.data.item_num])
        return user_all_embeddings, item_all_embeddings

    def cal_cl_loss(self, idx, perturbed_mat1, perturbed_mat2):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        user_view_1, item_view_1 = self.forward(perturbed_mat1)
        user_view_2, item_view_2 = self.forward(perturbed_mat2)
        view1 = torch.cat((user_view_1[u_idx],item_view_1[i_idx]),0)
        view2 = torch.cat((user_view_2[u_idx],item_view_2[i_idx]),0)
        return InfoNCE(view1,view2,self.temp)


class GraphGenerator(nn.Module):
    def __init__(self, data, emb_size):
        super(GraphGenerator, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.predictor = nn.Sequential(
            nn.Linear(self.latent_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1),
            #nn.Dropout(p=0.5),
            nn.Tanh()
        )

    def forward(self, noise):
        return self.predictor(noise)

class GraphGenerator_MF(nn.Module):
    def __init__(self, data, emb_size):
        super(GraphGenerator_MF, self).__init__()
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

class GraphGenerator_MLP(nn.Module):
    def __init__(self, data, emb_size):
        super(GraphGenerator_MLP, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.predictor = nn.Sequential(
            nn.Linear(self.latent_size*2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, user_e, item_e):
        return self.predictor(torch.cat((user_e,item_e), axis = 1))

class GraphGenerator_VAE(nn.Module):
    def __init__(self, data, emb_size):
        super(GraphGenerator_VAE, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.encoder = nn.Linear(self.latent_size*2, 64)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.fc_encoder_mu = nn.Linear(64, 16)
        self.fc_encoder_var = nn.Linear(64, 16)
        self.fc_reparameterize = nn.Linear(16, 64)
        self.fc_decode = nn.Linear(64, 1)

    def encode(self, x):
        output = self.encoder(x)
        h = self.relu(output)
        return self.fc_encoder_mu(h), self.fc_encoder_var(h)
 
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.rand_like(std)
        return mu + eps * std
 
    def decode(self, z):
        h = self.relu(self.fc_reparameterize(z))
        return self.sigmoid(self.fc_decode(h))

    def forward(self, user_e, item_e):
        input_vec = torch.cat((user_e,item_e), axis = 1)
        mu, log_var = self.encode(input_vec)
        z = self.reparameterize(mu, log_var)
        return self.decode(z)

