'''
Created on Nov 10, 2017
Create Model

@author: Lianhai Miao
'''

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from utils import ReverseLayerF
import itertools
import time

class AGREE_trans(nn.Module):
    def __init__(self, num_users, num_items, num_groups, num_users_scr, num_items_scr, embedding_dim, group_member_dict, device, drop_ratio, lmd, eta):
        super(AGREE_trans, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_groups = num_groups
        self.num_users_ = num_users_scr
        self.num_items = num_items_scr

        self.embedding_dim = embedding_dim
        # self.embedding_out = embedding_out
        self.group_member_dict = group_member_dict
        self.device = device
        self.drop_ratio = drop_ratio
        self.lmd = lmd
        self.eta = eta

        self.userembeds = nn.Embedding(num_users, embedding_dim) 
        self.itemembeds = nn.Embedding(num_items, embedding_dim)
        self.groupembeds = nn.Embedding(num_groups, embedding_dim)
        self.userembeds_scr = nn.Embedding(num_users_scr, embedding_dim) 
        self.itemembeds_scr = nn.Embedding(num_items_scr, embedding_dim)

        self.attention = AttentionLayer(2 * embedding_dim, drop_ratio)
        self.attention_u = AttentionLayer(embedding_dim, drop_ratio)
        
        self.self_attention_tuser = SelfAttentionLayer_tuser(embedding_dim, drop_ratio)
        
        self.pred_domain_1 = PredDomainLayer_1(3 * embedding_dim)
        self.pred_domain = PredDomainLayer(embedding_dim)
        self.predictlayer_gro = PredictLayer(3 * embedding_dim, drop_ratio)
        self.predictlayer_u = PredictLayer(3 * embedding_dim, drop_ratio)
        
        self.fcl = nn.Linear(self.embedding_dim, 1)
        
        # initial model
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
            if isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)


    def forward(self, user_inputs, item_inputs, type_m, source=True, p=1):
        if type_m == 'group':
            # get the item and group full embedding vectors
            item_embeds_full = self.itemembeds(item_inputs)
            group_embeds_full = self.groupembeds(user_inputs)

            g_embeds_with_attention = torch.zeros([len(user_inputs), self.embedding_dim])

            user_ids = [self.group_member_dict[usr.item()] for usr in user_inputs]
            MAX_MENBER_SIZE = max([len(menb) for menb in user_ids])
            menb_ids, item_ids, mask = [None]*len(user_ids),  [None]*len(user_ids),  [None]*len(user_ids)
            for i in range(len(user_ids)):
                postfix = [0]*(MAX_MENBER_SIZE - len(user_ids[i]))
                menb_ids[i] = user_ids[i] + postfix
                item_ids[i] = [item_inputs[i].item()]*len(user_ids[i]) + postfix
                mask[i] = [1]*len(user_ids[i]) + postfix
            
            menb_ids, item_ids, mask = torch.Tensor(menb_ids).long().to(self.device),\
                                        torch.Tensor(item_ids).long().to(self.device),\
                                        torch.Tensor(mask).float().to(self.device)
            
            menb_emb =  self.userembeds(menb_ids)
            menb_emb *= mask.unsqueeze(dim=-1) # [B, N, C] 
            item_emb = self.itemembeds(item_ids) # [B, N, C] 
            item_emb *= mask.unsqueeze(dim=-1)
            group_item_emb = torch.cat((menb_emb, item_emb), dim=-1) # [B, N, 2C], N=MAX_MENBER_SIZE
            # group_item_emb = group_item_emb.view(-1, group_item_emb.size(-1)) # [B * N, 2C]
            attn_weights = self.attention(group_item_emb)# [B * N, 1]
            attn_weights = attn_weights.squeeze(dim=-1)
            # attn_weights = attn_weights.view(menb_ids.size(0), -1) # [B, N] 
            attn_weights_exp = attn_weights.exp() * mask
            attn_weights_sm = attn_weights_exp/torch.sum(attn_weights_exp, dim=-1, keepdim=True) # [B, N] 
            attn_weights_sm = attn_weights_sm.unsqueeze(dim=1) # [B, 1, N]
            g_embeds_with_attention = torch.bmm(attn_weights_sm, menb_emb) # [B, 1, C]
            g_embeds_with_attention = g_embeds_with_attention.squeeze(dim=1)
            # print(time.time() - start)
            
            # put the g_embeds_with_attention matrix to GPU
            g_embeds_with_attention = g_embeds_with_attention.to(self.device)
            
            # obtain the group embedding which consists of two components: user embedding aggregation and group preference embedding
            g_embeds = g_embeds_with_attention + group_embeds_full
            # g_embeds = torch.add(torch.mul(g_embeds_with_attention, 0.7), torch.mul(group_embeds_full, 0.3))
                
            element_embeds = torch.mul(g_embeds, item_embeds_full)  # Element-wise product
            new_embeds = torch.cat((element_embeds, g_embeds, item_embeds_full), dim=1)
            preds_gro = torch.sigmoid(self.predictlayer_gro(new_embeds))

            return preds_gro

        elif type_m == 'sa_group':
            # get the item and group full embedding vectors
            item_embeds_full = self.itemembeds(item_inputs)   # [B, C]
            group_embeds_full = self.groupembeds(user_inputs) # [B, C]

            g_embeds_with_attention = torch.zeros([len(user_inputs), self.embedding_dim])
            # g_embeds_with_self_attention = torch.zeros([len(user_inputs), self.embedding_dim])

            user_ids = [self.group_member_dict[usr.item()] for usr in user_inputs] # [B,1]
            MAX_MENBER_SIZE = max([len(menb) for menb in user_ids]) # the great group size = 4
            menb_ids, item_ids, mask = [None]*len(user_ids),  [None]*len(user_ids),  [None]*len(user_ids)
            for i in range(len(user_ids)):
                postfix = [0]*(MAX_MENBER_SIZE - len(user_ids[i]))
                menb_ids[i] = user_ids[i] + postfix
                item_ids[i] = [item_inputs[i].item()]*len(user_ids[i]) + postfix
                mask[i] = [1]*len(user_ids[i]) + postfix
            
            menb_ids, item_ids, mask = torch.Tensor(menb_ids).long().to(self.device),\
                                        torch.Tensor(item_ids).long().to(self.device),\
                                        torch.Tensor(mask).float().to(self.device)

            
            menb_emb =  self.userembeds(menb_ids) # [B, N, C] 
            menb_emb *= mask.unsqueeze(dim=-1) # [B, N, C] 
            item_emb = self.itemembeds(item_ids) # [B, N, C] 
            item_emb *= mask.unsqueeze(dim=-1) # [B, N, C]
            
            #######################################
            ### Self-attention layer 
            #######################################
            proj_query_emb, proj_key_emb, proj_value_emb = self.self_attention_tuser(menb_emb) # [B, N, C/2], [B, N, C/2], [B, N, C]
            proj_query_emb_new = proj_query_emb * mask.unsqueeze(dim=-1)
            proj_key_emb_new = proj_key_emb * mask.unsqueeze(dim=-1)
            energy = torch.bmm(proj_query_emb_new, proj_key_emb_new.permute(0,2,1)) # [B, N , N]

            energy_exp = energy.exp() * mask.unsqueeze(dim=1)

            energy_exp_softmax = energy_exp/torch.sum(energy_exp, dim=-1, keepdim=True) # [B, N, N]
            # if energy_exp_softmax.shape[1] <= 10:
            #     np.savetxt('D:/Desktop/DSS/ruxia/AGREE_ruxia/Experiments/MaFengWo/SAGREE_trans/energy_exp_softmax', energy_exp_softmax.cpu().detach().numpy()[0], fmt='%1.4f', delimiter=' ') 
            menb_emb_out = torch.bmm(energy_exp_softmax, proj_value_emb) # [B, N, N] * [B, N, C] = [B, N, C]
            menb_emb_out_new = menb_emb_out * mask.unsqueeze(dim=-1)
            overall_menb_out = menb_emb_out_new + menb_emb # [B, N, C]


            ##########################################
            ### Vanilla attention layer 
            ##########################################
            group_item_emb = torch.cat((overall_menb_out, item_emb), dim=-1) # [B, N, 2C], N=MAX_MENBER_SIZE
            attn_weights = self.attention(group_item_emb)# [B, N, 1]
            # group_item_emb = overall_menb_out # [B, N, C]
            # attn_weights = self.attention_u(group_item_emb)# [B, N, 1]
            attn_weights = attn_weights.squeeze(dim=-1) # [B, N]
            attn_weights_exp = attn_weights.exp() * mask # [B, N]
            attn_weights_sm = attn_weights_exp/torch.sum(attn_weights_exp, dim=-1, keepdim=True) # [B, N] 

            # if energy_exp_softmax.shape[1] <= 10:
            #     np.savetxt('D:/Desktop/DSS/ruxia/AGREE_ruxia/Experiments/MaFengWo/SAGREE_trans/energy_exp_softmax', energy_exp_softmax.cpu().detach().numpy()[0], fmt='%1.4f', delimiter=' ')
            #     np.savetxt('D:/Desktop/DSS/ruxia/AGREE_ruxia/Experiments/MaFengWo/SAGREE_trans/att_groups', attn_weights_sm.cpu().detach().numpy()[:10], fmt='%1.4f', delimiter=' ')
            
            attn_weights_sm = attn_weights_sm.unsqueeze(dim=1) # [B, 1, N]
            
            g_embeds_with_attention = torch.bmm(attn_weights_sm, overall_menb_out) # [B, 1, C]
            g_embeds_with_attention = g_embeds_with_attention.squeeze(dim=1)

            
            # put the g_embeds_with_attention matrix to GPU
            g_embeds_with_attention = g_embeds_with_attention.to(self.device)

            # obtain the group embedding which consists of two components: user embedding aggregation and group preference embedding
            g_embeds = self.lmd * g_embeds_with_attention + group_embeds_full         # HAN-CDGR
            # g_embeds = self.lmd * g_embeds_with_attention                               # uH
            # g_embeds = group_embeds_full                                              # G
                
            element_embeds = torch.mul(g_embeds, item_embeds_full)  # Element-wise product
            preds_gro = torch.sigmoid(self.fcl(element_embeds))

            #new_embeds = torch.cat((element_embeds, g_embeds, item_embeds_full), dim=1)
            #preds_gro = torch.sigmoid(self.predictlayer_gro(new_embeds))
            
            return preds_gro

        elif type_m == 'G_group':
            # get the item and group full embedding vectors
            item_embeds_full = self.itemembeds(item_inputs)   # [B, C]
            group_embeds_full = self.groupembeds(user_inputs) # [B, C]

            element_embeds = torch.mul(group_embeds_full, item_embeds_full)  # Element-wise product
            preds_gro = torch.sigmoid(self.fcl(element_embeds))

            #new_embeds = torch.cat((element_embeds, g_embeds, item_embeds_full), dim=1)
            #preds_gro = torch.sigmoid(self.predictlayer_gro(new_embeds))
            
            return preds_gro
        

        elif type_m == 'target_user_HA':
            # get the target user and item embedding vectors
            user_embeds = self.userembeds(user_inputs)
            item_embeds = self.itemembeds(item_inputs)

            # start=time.time()
            # get the group id (key) of the user_inputs and then get all the group member ids(group_user_ids) in the group
            # Get each user_inputs' keys(groups) in the self.group_menb_dict, Note! one user_input may belong to more than one group!!!
            user_inputs_keys = [self.get_keys(self.group_member_dict, usr.item()) for usr in user_inputs]

            new_user_inputs = [None] * len(user_inputs)
            for i in range(len(user_inputs)):
                new_user_inputs[i] = [user_inputs[i]] * len(user_inputs_keys[i]) 

            new_user_inputs = [usr for u in new_user_inputs for usr in u] # flatten the nested list new_user_inputs  length = X
            new_user_inputs = torch.Tensor(new_user_inputs).long().to(self.device) # shape: (X,)
            new_user_embeds = self.userembeds(new_user_inputs) # shape:[X, C]

            group_input = [group_id for group in user_inputs_keys for group_id in group] # flatten the user_inputs_keys which is a nested list
            group_user_ids = [self.group_member_dict[k] for k in group_input] # length = X

            # get the great group size
            MAX_MENBER_SIZE = max([len(menb) for menb in group_user_ids]) # the great group size = 4
            # menb_ids is group members and empty members, mask1 is to mask the empty members, mask is to mask all the other members that is not the user_input id
            menb_ids, mask1 = [None]*len(group_user_ids),  [None]*len(group_user_ids)
            for i in range(len(group_user_ids)):
                postfix = [0]*(MAX_MENBER_SIZE - len(group_user_ids[i])) 
                menb_ids[i] = group_user_ids[i] + postfix

                mask1[i] = [1]*len(group_user_ids[i]) + postfix

            menb_ids, mask1 = torch.Tensor(menb_ids).long().to(self.device),\
                                        torch.Tensor(mask1).float().to(self.device)
            # [X,N] : menb_ids, mask1
            
            mask = (menb_ids == new_user_inputs.unsqueeze(-1)).float().to(self.device) # [X, N]
            # mask = torch.where()

            # Get the menb_emb
            menb_emb =  self.userembeds(menb_ids) # [B, N, C] 
            menb_emb *= mask1.unsqueeze(dim=-1) # [B, N, C] * [B,N,1] = [B,N,C] Turn the empty menber rows into empty rows
            
            ################################
            ## Self-attention part #########
            ################################
            proj_query_emb, proj_key_emb, proj_value_emb = self.self_attention_tuser(menb_emb) # [B, N, C/2], [B, N, C/2], [B, N, C]
            proj_query_emb_new = proj_query_emb * mask1.unsqueeze(dim=-1)
            proj_key_emb_new = proj_key_emb * mask1.unsqueeze(dim=-1)
            energy = torch.bmm(proj_query_emb_new, proj_key_emb_new.permute(0,2,1))/torch.sqrt(torch.tensor(menb_emb.shape[-1], dtype=torch.float32)) # [B, N , N]

            energy_exp = energy.exp() * mask1.unsqueeze(dim=1)

            energy_exp_softmax = energy_exp/torch.sum(energy_exp, dim=-1, keepdim=True) # [B, N, N] 
            
            menb_emb_out = torch.bmm(energy_exp_softmax, proj_value_emb) # [G, N, N] * [G, N, C] = [B, N, C]
            menb_emb_out_new = menb_emb_out * mask1.unsqueeze(dim=-1) # [B,N,C]
            user_emb_out = menb_emb_out_new * mask.unsqueeze(-1) # [B,N,C] * [B,N,1] = [B,N,C]
            user_emb_out_new = torch.sum(user_emb_out, dim=1) # collapse the rows of user_emb_out and get a [B, C] matrix
            overall_user_emb_out = user_emb_out_new + new_user_embeds # shape: [X, C]

            #############################
            ## Vanilla attention part ###
            #############################
            attn_weights = self.attention_u(overall_user_emb_out)# [X, 1]

            # # Get a mask matrix to detect which user has joined more than one group
            # mask2 = [None] * len(user_inputs)
            # for i in range(len(user_inputs)):
            #     mask2[i] = torch.mul((new_user_inputs == user_inputs[i]).float(), 1./torch.tensor([len(menb) for menb in group_user_ids],dtype=torch.float32).to(self.device))

            # mask2_tensor = torch.stack(mask2) # [B, X]
            # mask2_sm = mask2_tensor/torch.sum(mask2_tensor, dim=-1, keepdim=True) # [B, X]

            # Get a mask matrix to detect which user has joined more than one group
            mask2 = [None] * len(user_inputs)
            for i in range(len(user_inputs)):
                mask2[i] = (new_user_inputs == user_inputs[i]).cpu().numpy()

            mask2 = torch.Tensor(mask2).float().to(self.device) # The shape of mask2: [B, X]
            # multiplies each element of mask2 with the corresponding element of the attn_weights
            new_mask2 = torch.mul(mask2, attn_weights.view(1, -1)) #[B, X]
            # softmax
            new_mask2_sm = new_mask2/torch.sum(new_mask2, dim=-1, keepdim=True)
            # get each user's integrated preference
            new_overall_user_emb_out = torch.mm(new_mask2_sm, overall_user_emb_out) # [B, X] * [X, C] = [B, C]

            overall_menb_out = self.eta * new_overall_user_emb_out + user_embeds # [B, C] HAN-CDGR
            # overall_menb_out = self.eta * new_overall_user_emb_out                 # [B, C] gH
            # overall_menb_out = user_embeds                                       # [B, C] U  
            
            ######################################
            #### Rating prediction layer
            ######################################
            element_embeds = torch.mul(overall_menb_out, item_embeds)  # Element-wise product
            # pooling layer
            new_embeds = torch.cat((element_embeds, overall_menb_out, item_embeds), dim=1)
            # rating prediction
            preds_r = torch.sigmoid(self.predictlayer_u(new_embeds))
            
            ######################################
            ##### Adversarial training layer
            ######################################
            domain_output = self.get_adversarial_result_1(new_embeds, source=source, p=p)

            return preds_r, domain_output

            
        elif type_m == 'source_user_1':
            # get the target user and item embedding vectors
            user_embeds_scr = self.userembeds_scr(user_inputs)
            item_embeds_scr = self.itemembeds_scr(item_inputs)

            
            element_embeds_scr = torch.mul(user_embeds_scr, item_embeds_scr).to(self.device)  # Element-wise product
            new_embeds_scr = torch.cat((element_embeds_scr, user_embeds_scr, item_embeds_scr), dim=1)
           
            preds_r_scr = torch.sigmoid(self.predictlayer_u(new_embeds_scr))
            # preds_r_scr = torch.sigmoid(self.fcl(element_embeds_scr))

            # get the binary cross entropy loss between target R users true labels 0 and their predicted domain labels
            domain_output_scr = self.get_adversarial_result_1(new_embeds_scr, source=source, p=p)
            # domain_output_scr = self.get_adversarial_result(element_embeds_scr, source=source, p=p)
            
            return preds_r_scr, domain_output_scr

        elif type_m == 'user':
            user_embeds = self.userembeds(user_inputs)
            item_embeds = self.itemembeds(item_inputs)
            element_embeds = torch.mul(user_embeds, item_embeds)  # Element-wise product
            new_embeds = torch.cat((element_embeds, user_embeds, item_embeds), dim=1)
            preds = torch.sigmoid(self.predictlayer_gro(new_embeds))
            
            return preds
    
    def get_adversarial_result_1(self, x, source=True, p=0.0):
            loss_fn = nn.BCELoss()
            
            if source:
                domain_label = torch.zeros(len(x)).long().to(self.device)
            else:
                domain_label = torch.ones(len(x)).long().to(self.device)
                
            # get the reversed feature
            x = ReverseLayerF.apply(x, p)

            domain_pred = self.pred_domain_1(x)
            loss_adv = loss_fn(domain_pred, domain_label.float().unsqueeze(dim=1))
            return loss_adv

    def get_adversarial_result(self, x, source=True, p=0.0):
            loss_fn = nn.BCELoss()
            
            if source:
                domain_label = torch.zeros(len(x)).long().to(self.device)
            else:
                domain_label = torch.ones(len(x)).long().to(self.device)
                
            # get the reversed feature
            x = ReverseLayerF.apply(x, p)

            domain_pred = self.pred_domain(x)
            
            loss_adv = loss_fn(domain_pred, domain_label.float().unsqueeze(dim=1))
            return loss_adv


    def get_keys(self, d, value):

        return [k for k, v in d.items() if value in v]



class AttentionLayer(nn.Module):
    """ Attention Layer"""
    def __init__(self, embedding_dim, drop_ratio=0):
        super(AttentionLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 16),
            nn.ReLU(),
            nn.Dropout(drop_ratio, self.training),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        out = self.linear(x)
        return out
        # weight = F.softmax(out.view(1, -1), dim=1)
        # return weight

class SelfAttentionLayer_tuser(nn.Module):
    """ Self attention Layer"""
    def __init__(self, embedding_dim, drop_ratio=0.1):
        super(SelfAttentionLayer_tuser, self).__init__()
        self.embedding_dim = embedding_dim

        self.query_linear = nn.Sequential()
        self.query_linear.add_module('fc_ise1_query', nn.Linear(embedding_dim, embedding_dim//2))
        self.query_linear.add_module('ac_ise1_query', nn.ReLU(True))
        self.query_linear.add_module('dropout_query', nn.Dropout(drop_ratio))

        self.key_linear = nn.Sequential()
        self.key_linear.add_module('fc_ise1_key', nn.Linear(embedding_dim, embedding_dim//2))
        self.key_linear.add_module('ac_ise1_key', nn.ReLU(True))
        self.key_linear.add_module('dropout_key', nn.Dropout(drop_ratio))

        self.value_linear = nn.Sequential()
        self.value_linear.add_module('fc_ise1_value', nn.Linear(embedding_dim, embedding_dim))
        self.value_linear.add_module('ac_ise1_value', nn.ReLU(True))
        self.value_linear.add_module('value_query', nn.Dropout(drop_ratio))

    def forward(self, x):
        """ 
            Inputs :
                x  : a group members' embeddings cat item embeddings [B, N, 2C]
            Returns :
                out : out : self attention value + input feature         
        """
        proj_query = self.query_linear(x) # [B, N , C//2]
        proj_key = self.key_linear(x) # [B, N , C//2]
        proj_value = self.value_linear(x) # [B, N , C]
        
        return proj_query, proj_key, proj_value


class SelfAttentionLayer_pre(nn.Module):
    """ Self attention Layer"""
    def __init__(self, embedding_dim):
        super(SelfAttentionLayer_pre, self).__init__()
        self.embedding_dim = embedding_dim

        self.query_linear = nn.Sequential()
        self.query_linear.add_module('fc_ise1', nn.Linear(embedding_dim, embedding_dim//2))
        self.query_linear.add_module('ac_ise1', nn.ReLU(True))

        self.key_linear = nn.Sequential()
        self.key_linear.add_module('fc_ise1', nn.Linear(embedding_dim, embedding_dim//2))
        self.key_linear.add_module('ac_ise1', nn.ReLU(True))

        # Noteworthy:
        # the dimension of query and key must always be the same because of the dot product score function
        # however, the dimension of value may be different from query and key.
        # the resulting output will consequently follow the dimension of value.

        self.value_linear = nn.Sequential()
        self.value_linear.add_module('fc_ise1', nn.Linear(embedding_dim, embedding_dim))
        self.value_linear.add_module('ac_ise1', nn.ReLU(True))

        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #

        
    def forward(self, x):
        """ 
            Inputs :
                x  : a group members' embeddings
            Returns :
                out : out : self attention value + input feature         
        """
        proj_query = self.query_linear(x) # members_size * embedding_dim//2
        proj_key = self.key_linear(x) # members_size * embedding_dim//2
        energy = torch.mm(proj_query, proj_key.t()) # members_size * members_size
        attention = self.softmax(energy) # members_size * members_size

        proj_value = self.value_linear(x) # members_size * embedding_dim
        out = torch.mm(attention, proj_value) # members_size * embedding_dim
        out = self.gamma*out + x

        out = torch.mean(out, dim=0).unsqueeze(dim=0) # 1 * embedding_dim

        return out


class PredDomainLayer_1(nn.Module):
    def __init__(self, embedding_dim):
        super(PredDomainLayer_1, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )

    def forward(self, input):
        out = self.linear(input)
        pred_label = torch.sigmoid(out)
       
        return pred_label



class PredDomainLayer(nn.Module):
    def __init__(self,embedding_dim):
        super(PredDomainLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )

    def forward(self, input):
        out = self.linear(input)
        pred_label = torch.sigmoid(out)
       
        return pred_label



class PredictLayer(nn.Module):
    def __init__(self, embedding_dim, drop_ratio=0):
        super(PredictLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 8),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        out = self.linear(x)
        return out

