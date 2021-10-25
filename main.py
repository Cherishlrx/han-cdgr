import faulthandler; faulthandler.enable()
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from time import time
from utils import Helper, AGREELoss
from dataset import GDataset
# from model_sa import AGREE_trans
from model import AGREE_trans
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--save_name', type=str, default='GU-CDGR_100')

parser.add_argument('--path', type=str, default='./Experiments/MaFengWo/HAN-CDGR/data/')
parser.add_argument('--user_dataset', type=str, default= './Experiments/MaFengWo/HAN-CDGR/data/' + 'userRating')
parser.add_argument('--group_dataset', type=str, default= './Experiments/MaFengWo/HAN-CDGR/data/' + 'groupRating')
parser.add_argument('--user_in_group_path', type=str, default= './Experiments/MaFengWo/HAN-CDGR/data/groupMember.txt')
parser.add_argument('--scr_user_dataset', type=str, default= './Experiments/MaFengWo/HAN-CDGR/data/' + 'yelpRatingTrain.txt')

# parser.add_argument('--path', type=str, default='./data/')
# parser.add_argument('--user_dataset', type=str, default= './data/' + 'userRating')
# parser.add_argument('--group_dataset', type=str, default= './data/' + 'groupRating')
# parser.add_argument('--user_in_group_path', type=str, default= './data/groupMember.txt')
# parser.add_argument('--scr_user_dataset', type=str, default= './data/' + 'yelpRatingTrain.txt')

parser.add_argument('--embedding_size_list', type=list, default=[32])
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--num_negatives', type=list, default=4)
parser.add_argument('--batch_size_list', type=list, default=[64])
parser.add_argument('--batch_size_user', type=int, default=64) 
parser.add_argument('--lr', type=list, default=[0.00002, 0.000005, 0.0000005])
# parser.add_argument('--lr', type=list, default=[0.001, 0.005, 0.05])
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--drop_ratio_list', type=list, default=[0.2])
parser.add_argument('--topK_list', type=list, default=[5])
parser.add_argument('--type_m_gro', type=str, default='G_group')
parser.add_argument('--type_m_usr', type=str, default='target_user_HA')
parser.add_argument('--type_m_scr', type=str, default='source_user_1')

parser.add_argument('--lmd_list', type=list, default=[0.6])
parser.add_argument('--eta_list', type=list, default=[0.3])

parser.add_argument('--gamma_weight_list', type=float, default=[0.005])
parser.add_argument('--beta_weight_list', type=float, default=[0.9])


args = parser.parse_args()

DEVICE = torch.device('cuda:{}'.format(args.cuda)) if torch.cuda.is_available() else torch.device('cpu')

# DEVICE = torch.device('cpu')

# train the model
def training_gro(model, train_loader, epoch_id, type_m, beta_weight):
    # user training
    learning_rates = args.lr

    # lr = learning_rates[0]
    # if epoch_id >= 5 and epoch_id < 10:
    #     lr = learning_rates[1]
    # elif epoch_id >=10:
    #     lr = learning_rates[2]
    # if epoch_id % 5 == 0:
    #     lr /= 2
    # learning rate decay
    lr = learning_rates[0]
    if epoch_id >= 20 and epoch_id < 60:
        lr = learning_rates[1]
    elif epoch_id >=60:
        lr = learning_rates[2]

    # if epoch_id >= 100 and epoch_id < 150:
    #     lr = learning_rates[1]
    # elif epoch_id >=150:
    #     lr = learning_rates[2]
    # # lr decay
    # if epoch_id % 50 == 0:
    #     lr /= 2

    # optimizer
    # optimizer = optim.RMSprop(model.parameters(), lr)
    optimizer = optim.Adam(model.parameters(), lr)
    # loss function
    loss_function = AGREELoss()
    losses = [] 
    model = model.to(DEVICE)    
    for batch_id, (u, pi_ni) in enumerate(train_loader):
        # Data Load
        user_input = u
        pos_item_input = pi_ni[:, 0]
        neg_item_input = pi_ni[:, 1]

        user_input, pos_item_input, neg_item_input = user_input.to(DEVICE), pos_item_input.to(DEVICE), neg_item_input.to(DEVICE)
        # Forward
        pos_prediction = model(user_input, pos_item_input, type_m)
        neg_prediction = model(user_input, neg_item_input, type_m)
        # Zero_grad
        model.zero_grad()
        # Loss value of one batch of examples
        loss = beta_weight * loss_function(pos_prediction, neg_prediction)
        # record loss history
        losses.append(loss)  
        # Backward
        loss.backward(torch.ones_like(loss))
        # updata parameters
        optimizer.step()

    print('Iteration %d, loss is [%.4f ]' % (epoch_id, torch.mean(torch.tensor(losses))))

def training_user(model, train_loader_r, train_loader_scr, epoch_id, beta_weight, gamma_weight):
    # user training
    learning_rates = args.lr
    # learning rate decay
    lr = learning_rates[0]
    if epoch_id >= 20 and epoch_id < 60:
        lr = learning_rates[1]
    elif epoch_id >=60:
        lr = learning_rates[2]
   
    # if epoch_id >= 60 and epoch_id < 100:
    #     lr = learning_rates[1]
    # elif epoch_id >=100:
    #     lr = learning_rates[2]

    # optimizer
    # optimizer = optim.RMSprop(model.parameters(), lr)
    optimizer = optim.Adam(model.parameters(), lr)
    # loss function
    loss_function = AGREELoss()
    len_dataloader = min(len(train_loader_r), len(train_loader_scr))

    losses_preds = [] 
    losses_domain_target = [] 
    losses_domain_source = []
    losses = []
    i = 1
    model = model.to(DEVICE)   
    for (data_usr, data_scr) in zip(enumerate(train_loader_r), enumerate(train_loader_scr)):
        # Data Load
        _, (usr, pi_ni_usr) = data_usr
        _, (usr_scr, pi_ni_scr) = data_scr

        user_input = usr
        pos_item_input = pi_ni_usr[:, 0]
        neg_item_input = pi_ni_usr[:, 1]

        user_input_scr = usr_scr
        pos_item_input_scr = pi_ni_scr[:, 0]
        neg_item_input_scr = pi_ni_scr[:, 1]


        user_input, pos_item_input, neg_item_input = user_input.to(DEVICE), pos_item_input.to(DEVICE), neg_item_input.to(DEVICE)
        user_input_scr, pos_item_input_scr, neg_item_input_scr = user_input_scr.to(DEVICE), pos_item_input_scr.to(DEVICE), neg_item_input_scr.to(DEVICE)
        
        p = float(i + epoch_id * len_dataloader) / args.n_epoch / len_dataloader
        p = 2. / (1. + np.exp(-10 * p)) - 1

        # Forward
        # target user forward
        pos_preds_usr, pos_domain_output = model(user_input, pos_item_input, args.type_m_usr, source=False, p=p)
        
        neg_preds_usr, neg_domain_output = model(user_input, neg_item_input, args.type_m_usr, source=False, p=p)

        # source user forward
        pos_preds_scr, pos_domain_output_scr = model(user_input_scr, pos_item_input_scr, args.type_m_scr, source=True, p=p)
        
        neg_preds_scr, neg_domain_output_scr = model(user_input_scr, neg_item_input_scr, args.type_m_scr, source=True, p=p)
    
        
        # Zero_grad
        model.zero_grad()
        # Loss value of one batch of examples
        loss_usr = loss_function(pos_preds_usr, neg_preds_usr)
        loss_scr = loss_function(pos_preds_scr, neg_preds_scr)

        # loss_domain = pos_domain_output_r + neg_domain_output_r + pos_domain_output_scr + neg_domain_output_scr
        loss_domain_target = (pos_domain_output + neg_domain_output)/2
        loss_domain_source = (pos_domain_output_scr + neg_domain_output_scr)/2

        loss_preds = beta_weight * (loss_usr + loss_scr)
        loss_domain = loss_domain_target + loss_domain_source

        loss = loss_preds + gamma_weight * loss_domain

        # record loss history
        losses_preds.append(loss_preds)
        losses_domain_target.append(loss_domain_target)
        losses_domain_source.append(loss_domain_source)

        losses.append(loss)

        # Backward
        loss.backward(torch.ones_like(loss))
        # updata parameters
        optimizer.step()

        i +=1

    print('Iteration %d, loss is [%.4f ], loss_preds is [%.4f ], loss_domain_target is [%.4f ], loss_domain_source is [%.4f ]' % (epoch_id, torch.mean(torch.tensor(losses)), torch.mean(torch.tensor(losses_preds)),
    torch.mean(torch.tensor(losses_domain_target)), torch.mean(torch.tensor(losses_domain_source)))
    )



def evaluation(model, helper, testRatings, testNegatives, K, type_m, DEVICE):
    model = model.to(DEVICE)
    # set the module in evaluation mode
    model.eval()
    (hits, ndcgs) = helper.evaluate_model(model, testRatings, testNegatives, K, type_m, DEVICE)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    return hr, ndcg


if __name__ == '__main__':
    torch.random.manual_seed(1314)
    # initial helper
    helper = Helper()

    # get the dict of users in group
    g_m_d = helper.gen_group_member_dict(args.user_in_group_path)
    
    # initial dataSet class
    dataset = GDataset(args.user_dataset, args.group_dataset, args.scr_user_dataset, args.num_negatives)

    # get group number
    num_groups = len(g_m_d)
    num_users, num_items = dataset.num_users, dataset.num_items
    num_users_scr, num_items_scr = dataset.num_users_scr, dataset.num_items_scr
    
    print('Data prepare is over!')

    save_name = os.path.basename(args.save_name)
    dir_name = os.path.dirname(args.path)
    dir_name = os.path.join(dir_name, save_name)

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    for topK in args.topK_list:
        for embedding_size in args.embedding_size_list:
            for beta_weight in args.beta_weight_list:
                for gamma_weight in args.gamma_weight_list:
                    for batch_size in args.batch_size_list:
                        for drop_ratio in args.drop_ratio_list:
                            for lmd in args.lmd_list:
                                for eta in args.eta_list:
                                    for i in range(5):
                                        # load the AGREE_trans model
                                        agree = AGREE_trans(num_users, num_items, num_groups, num_users_scr, num_items_scr, embedding_size, g_m_d, DEVICE, drop_ratio, lmd, eta).to(DEVICE)
                                        
                                        # model_save_path = os.path.join(args.path, 'model_2.pth')
                                        # agree.load_state_dict(torch.load(model_save_path))
                                        # agree.eval()

                                        # args information
                                        print("AGREE_trans at embedding size %d, beta %1.3f, gamma %1.3f, batch_size %d, run Iteration:%d, NDCG and HR at %d, drop_ratio at %1.2f, lmd at %1.2f, eta at %1.2f" %(embedding_size, 
                                        beta_weight, gamma_weight, batch_size, args.n_epoch, topK, drop_ratio, lmd, eta))

                                        # train the model
                                        HR_gro = []
                                        NDCG_gro = []
                                        HR_user = []
                                        NDCG_user = []
                                        user_train_time = []
                                        gro_train_time = []
                                        # for epoch in range(args.n_epoch):
                                        for epoch in range(args.n_epoch):
                                            # set the module in training mode
                                            agree.train()
                                            # 开始训练时间
                                            t1_user = time()
                                            # pretrain the target user and source user
                                            training_user(agree, dataset.get_user_dataloader(args.batch_size_user), dataset.get_scr_dataloader(args.batch_size_user), epoch, beta_weight, gamma_weight)
                                            print("user training time is: [%.1f s]" % (time()-t1_user))
                                            user_train_time.append(time()-t1_user)

                                            t1_gro = time()
                                            # retrain the group
                                            training_gro(agree, dataset.get_group_dataloader(batch_size), epoch, args.type_m_gro, beta_weight)                                               
                                            print("group training time is: [%.1f s]" % (time()-t1_gro))
                                            gro_train_time.append(time()-t1_gro)

                                            # evaluation
                                            t2 = time()
                                            u_hr, u_ndcg = evaluation(agree, helper, dataset.user_testRatings, dataset.user_testNegatives, topK, args.type_m_usr, DEVICE)
                                            HR_user.append(u_hr)
                                            NDCG_user.append(u_ndcg)
                                            
                                            print('User Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, [%.1f s]' % (
                                                epoch, time() - t1_user, u_hr, u_ndcg, time() - t2))

                                            t3 = time()
                                            hr, ndcg = evaluation(agree, helper, dataset.group_testRatings, dataset.group_testNegatives, topK, args.type_m_gro, DEVICE)
                                            HR_gro.append(hr)
                                            NDCG_gro.append(ndcg)
                                            print(
                                                'Group Iteration %d [%.1f s]: HR = %.4f, '
                                                'NDCG = %.4f, [%.1f s]' % (epoch, time() - t1_user, hr, ndcg, time() - t3))

                                            # save the model on GPU
                                            # if (epoch >0) and (epoch % 10 == 0):
                                            #     model_save_path = os.path.join(args.path, 'model_{}.pth'.format(epoch))
                                            #     torch.save(agree.state_dict(), model_save_path)

                                            #     print('model_{} has been saved'.format(epoch))


                                        # EVA_user = np.column_stack((HR_user, NDCG_user))
                                        # EVA_gro = np.column_stack((HR_gro, NDCG_gro, gro_train_time))

                                        EVA_data = np.column_stack((HR_user, NDCG_user, user_train_time, HR_gro, NDCG_gro, gro_train_time))

                                        print("save to file...")

                                        filename = "EVA_%s_%s_E%d_beta%1.3f_gamma%1.3f_batch%d_topK%d_drop_ratio%1.2f_lambda_%1.2f_eta_%1.2f_%d" % (args.type_m_gro, args.type_m_usr, embedding_size, beta_weight, gamma_weight, batch_size, topK, drop_ratio, lmd, eta, i)

                                        filename = os.path.join(dir_name, filename)

                                        np.savetxt(filename, EVA_data, fmt='%1.4f', delimiter=' ')

                                        print("Done!")
