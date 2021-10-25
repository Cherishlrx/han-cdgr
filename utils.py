'''
Created on Nov 10, 2017
Deal something

@author: Lianhai Miao
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Function
import numpy as np
import math
import heapq

# 参考网址 https://github.com/jindongwang/transferlearning/blob/master/code/deep/DANN(RevGrad)/adv_layer.py
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, p):
        ctx.p = p

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.p

        return output, None


class AGREELoss(nn.Module):
    def __init__(self):
        super(AGREELoss, self).__init__()
    
    def forward(self, pos_preds, neg_preds):
        
        loss = torch.mean((pos_preds - neg_preds - 1).clone().pow(2))

        return loss

class Helper(object):
    """
        utils class: it can provide any function that we need
    """
    def __init__(self):
        self.timber = True

    def gen_group_member_dict(self, path):
        g_m_d = {}
        with open(path, 'r') as f:
            line = f.readline().strip()
            while line != None and line != "":
                a = line.split(' ')
                g = int(a[0])
                g_m_d[g] = []
                for m in a[1].split(','):
                    g_m_d[g].append(int(m))
                line = f.readline().strip()
        return g_m_d
 

    # The following functions are used to evaluate NCF_trans and group recommendation performance
    def evaluate_model(self, model, testRatings, testNegatives, K, type_m, device):
        """
        Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
        Return: score of each test rating.
        """
        hits, ndcgs = [], []

        for idx in range(len(testRatings)):
            (hr,ndcg) = self.eval_one_rating(model, testRatings, testNegatives, K, type_m, idx, device)
            hits.append(hr)
            ndcgs.append(ndcg)

        return (hits, ndcgs)


    def eval_one_rating(self, model, testRatings, testNegatives, K, type_m, idx, device):
        p = 0
        rating = testRatings[idx]
        items = testNegatives[idx]
        u = rating[0]
        gtItem = rating[1]
        items.append(gtItem)
        # Get prediction scores
        map_item_score = {}
        users = np.full(len(items), u)

        users_var = torch.from_numpy(users).long().to(device)
        items_var = torch.LongTensor(items).to(device)

        if type_m == 'group':
            predictions = model(users_var, items_var, 'group')
        elif type_m == 'sa_group':
            predictions = model(users_var, items_var, 'sa_group')
        elif type_m == 'group_fixed_agg':
            predictions = model(users_var, items_var, 'group_fixed_agg')
        elif type_m == 'G_group':
            predictions = model(users_var, items_var, 'G_group')   
        # concat <user*item, user, item> and feed into Discriminator    
        elif type_m == 'target_user_HA':
            predictions, _ = model(users_var, items_var, 'target_user_HA', source=False, p=p)
        # feed user and item into Discriminator 
        elif type_m == 'source_user_1':
            predictions, _ = model(users_var, items_var, 'source_user_1', source=True, p=p)

          
        for i in range(len(items)):
            item = items[i]
            map_item_score[item] = predictions.cpu().data.numpy()[i]
        items.pop()

        # Evaluate top rank list
        ranklist = heapq.nlargest(K, map_item_score, key=map_item_score.get)
        hr = self.getHitRatio(ranklist, gtItem)
        ndcg = self.getNDCG(ranklist, gtItem)
        return (hr, ndcg)

    def getHitRatio(self, ranklist, gtItem):
        for item in ranklist:
            if item == gtItem:
                return 1
        return 0

    def getNDCG(self, ranklist, gtItem):
        for i in range(len(ranklist)):
            item = ranklist[i]
            if item == gtItem:
                return math.log(2) / math.log(i+2)
        return 0