'''
Created on Aug 8, 2016
Processing datasets.

@author: Xiangnan He (xiangnanhe@gmail.com)

Modified  on Nov 10, 2017, by Lianhai Miao
'''

import scipy.sparse as sp
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

class GDataset(object):

    def __init__(self, user_path, group_path, scr_path, num_negatives):
        '''
        Constructor
        '''
        self.num_negatives = num_negatives
        # user data
        self.user_trainMatrix = self.load_rating_file_as_matrix(user_path + "Train.txt")
        self.user_testRatings = self.load_rating_file_as_list(user_path + "Test.txt")
        self.user_testNegatives = self.load_negative_file(user_path + "Negative.txt")
        self.num_users, self.num_items = self.user_trainMatrix.shape
        # group data
        self.group_trainMatrix = self.load_rating_file_as_matrix(group_path + "Train.txt")
        self.group_testRatings = self.load_rating_file_as_list(group_path + "Test.txt")
        self.group_testNegatives = self.load_negative_file(group_path + "Negative.txt")

        # source user data
        self.scr_user_trainMatrix = self.load_rating_file_as_matrix(scr_path)
        
        self.num_users_scr, self.num_items_scr = self.scr_user_trainMatrix.shape



    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(" ")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        return ratingList

    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(" ")
                negatives = []
                for x in arr[1:]:
                    negatives.append(int(x))
                # negatives = [x for x in arr[1:]]
                negativeList.append(negatives)
                line = f.readline()
        return negativeList

    def load_rating_file_as_matrix(self, filename):
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                if filename=='./data/yelpRatingTrain.txt' or filename=='./data/HAN-CDGR-revision/Experiments/MaFengWo/SAGREE_trans/data/yelpRatingTrain.txt':
                    arr = line.split("\t")
                else:
                    arr = line.split(" ")

                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u) 
                num_items = max(num_items, i) 
                line = f.readline()
        # Construct matrix 
        mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                if (filename=='./data/yelpRatingTrain.txt') or (filename=='./data/HAN-CDGR-revision/Experiments/MaFengWo/SAGREE_trans/data/yelpRatingTrain.txt'):
                    arr = line.split("\t")
                else:
                    arr = line.split(" ")
                if len(arr) > 2:
                    # user, item, rating = int(arr[0]), int(arr[1]), int(arr[2])
                    user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                    if (rating > 0):
                        mat[user, item] = 1.0
                else:
                    user, item = int(arr[0]), int(arr[1])
                    mat[user, item] = 1.0
                line = f.readline()
        return mat

    def get_train_instances(self, train):
        user_input, pos_item_input, neg_item_input = [], [], []
        num_users = train.shape[0]
        num_items = train.shape[1]
        for (u, i) in train.keys():
            # positive instance
            for _ in range(self.num_negatives):
                pos_item_input.append(i)
            # negative instances
            for _ in range(self.num_negatives):
                j = np.random.randint(num_items)
                while (u, j) in train:
                    j = np.random.randint(num_items)
                user_input.append(u)
                neg_item_input.append(j)
        pi_ni = [[pi, ni] for pi, ni in zip(pos_item_input, neg_item_input)]
        return user_input, pi_ni

    def get_user_dataloader(self, batch_size):
        user, positem_negitem_at_u = self.get_train_instances(self.user_trainMatrix)
        
        train_data = TensorDataset(torch.LongTensor(user), torch.LongTensor(positem_negitem_at_u))
        user_train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, drop_last=True)
        return user_train_loader
    
    def get_group_dataloader(self, batch_size):
        # group and positem_negitem_at_g are two lists
        group, positem_negitem_at_g = self.get_train_instances(self.group_trainMatrix)
        # group is a list of group ids, positem_negitem_at_g is a list of list, whose elements are a list of positem id and negitem id
        # user_ids = [self.group_member_dict[group_id] for group_id in group] # user_ids is a list of list, whose members is a list of group member ids

        # MAX_MENBER_SIZE = max([len(menb) for menb in user_ids]) # the max group size = 4


        # menb_ids, pos_item_ids, neg_item_ids, mask = [None]*len(user_ids),  [None]*len(user_ids),  [None]*len(user_ids), [None]*len(user_ids)

        # for i in range(len(user_ids)):
        #     postfix = [0]*(MAX_MENBER_SIZE - len(user_ids[i]))
        #     menb_ids[i] = user_ids[i] + postfix
        #     pos_item_ids[i] = [positem_negitem_at_g[i][0]]*len(user_ids[i]) + postfix
        #     neg_item_ids[i] = [positem_negitem_at_g[i][1]]*len(user_ids[i]) + postfix
        #     mask[i] = [1]*len(user_ids[i]) + postfix
        
        train_data = TensorDataset(torch.LongTensor(group), torch.LongTensor(positem_negitem_at_g))
        # train_data = TensorDataset(torch.LongTensor(menb_ids), torch.LongTensor(pos_item_ids), torch.LongTensor(neg_item_ids),torch.LongTensor(mask))
        group_train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, drop_last=True)

        return group_train_loader

    def get_scr_dataloader(self, batch_size):
        # group and positem_negitem_at_g are two lists
        scr_user, positem_negitem_at_scr = self.get_train_instances(self.scr_user_trainMatrix)
            
        train_data = TensorDataset(torch.LongTensor(scr_user), torch.LongTensor(positem_negitem_at_scr))
        scr_train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        
        return scr_train_loader






