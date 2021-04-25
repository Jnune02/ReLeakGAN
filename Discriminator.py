#/////////////////
#/ Discriminator File
#///////////////

import torch
from scipy.stats import truncnorm
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import init_vars
from torch.autograd import Variable

#A truncated distribution has its domain (the x-values) restricted to a certain range of values. For example, you might restrict your x-values to between 0 and 100, written in math terminology as {0 > x > 100}. There are several types of truncated distributions:
def truncated_normal(shape, lower=-0.2, upper=0.2):
    size = 1
    for dim in shape:
        size *= dim
    w_truncated = truncnorm.rvs(lower, upper, size=size)
    w_truncated = torch.from_numpy(w_truncated).float()
    w_truncated = w_truncated.view(shape)
    return w_truncated
#!!

class Highway(nn.Module):
    #Highway Networks = Gating Function To Highway = y = xA^T + b
    def __init__(self, in_size, out_size):
        super(Highway, self).__init__()
        self.fc1 = nn.Linear(in_size, out_size)
        self.fc2 = nn.Linear(in_size, out_size)
    def forward(self, x):
        #highway = F.sigmoid(highway)*F.relu(highway) + (1. - transform)*pred # sets C = 1 - T
        g = F.relu(self.fc1)
        t = torch.sigmoid(self.fc2)
        out = g*t + (1. - t)*x
        return out

class Discriminator_l2(nn.Module):
    def __init__(self, seq_len, num_classes, vocab_size, dis_emb_dim, 
                    filter_sizes, num_filters, start_token, goal_out_size, step_size, dropout_prob, l2_reg_lambda):
        super(Discriminator_l2, self).__init__()
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.dis_emb_dim = dis_emb_dim
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.start_token = start_token
        self.goal_out_size = goal_out_size
        self.step_size = step_size
        self.dropout_prob = dropout_prob
        self.l2_reg_lambda = l2_reg_lambda
        self.num_filters_total = sum(self.num_filters)
        self.emb = nn.Embedding(self.vocab_size + 1, self.dis_emb_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_f, (f_size, self.dis_emb_dim)) for f_size, num_f in zip(self.filter_sizes, self.num_filters)
        ])
        self.highway = nn.Linear(self.num_filters_total, self.num_filters_total)
        #in_features = out_features = sum of num_festures
        self.dropout = nn.Dropout(p = self.dropout_prob)
        #Randomly zeroes some of the elements of the input tensor with probability p using Bernouli distribution
        #Each channel will be zeroed independently on every forward call
        self.fc = nn.Linear(self.num_filters_total, self.num_classes)

    def forward(self, x):
        """
        x: shape(batch_size * par_length_in_sentences * vector_data)
        type(torch.LongTensor)
        """
            
        convs = [F.relu(conv(x)).squeeze(3) for conv in self.convs] # [batch_size * num_filter * seq_len] --> seq_length: Number of sentences in padded paragraph.
        pooled_out = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs] # [batch_size * num_filter]
        pred = torch.cat(pooled_out, 1) # batch_size * sum(num_filters)
            #print("Pred size: {}".format(pred.size()))
        highway = self.highway(pred)
            #print("highway size: {}".format(highway.size()))
        highway = torch.sigmoid(highway)* F.relu(highway) + (1.0 - torch.sigmoid(highway))*pred
        features = self.dropout(highway)
        score = self.fc(features)
        pred = F.log_softmax(score, dim=1) #batch * num_classes
        return {"pred":pred, "feature":features, "score": score}

    def get_sample(self, model_dict, use_cuda=False, temperature=1.0):
        generator = model_dict["generator"]
        discriminator = model_dict["discriminator"]
        h_w_t, c_w_t, h_m_t, c_m_t, last_goal, real_goal, x_t = \
            init_vars(generator, discriminator, use_cuda)
        t = 0
        gen_token_list = []
        batch_size = generator.worker.batch_size
        seq_len = discriminator.seq_len
        step_size = generator.step_size
        goal_out_size = generator.worker.goal_out_size
        vocab_size = discriminator.vocab_size

        ##################################################################
            
        vector_size = discriminator.num_filters_total

        ##################################################################
            
        #G forward
        while t < seq_len:
            #Extract f_t
            if t == 0:
                cur_sen = Variable(nn.init.constant_(
                    torch.zeros(batch_size, seq_len, vector_size), vocab_size)
                ).long()
                
                if use_cuda:
                    cur_sen = cur_sen.cuda(non_blocking=True)
            else:
                cur_sen = torch.stack(gen_token_list).permute(1, 0)
                cur_sen = F.pad(
                    cur_sen, (0, seq_len - t), value=vocab_size
                )
            f_t = discriminator(cur_sen)["feature"]
            #G forward step
            x_t, h_m_t, c_m_t, h_w_t, c_w_t, last_goal, real_goal, sub_goal, probs, t_ = generator(x_t, f_t, h_m_t, c_m_t, h_w_t, c_w_t, last_goal,real_goal, t, temperature)
            if t % step_size == 0:
                if t > 0:
                    real_goal = last_goal
                    last_goal = Variable(torch.zeros(batch_size, goal_out_size))
                if use_cuda:
                    last_goal = last_goal.cuda(non_blocking=True)
                gen_token_list.append(x_t)
                t = t_
            gen_token = torch.stack(gen_token_list).permute(1,0)
        return gen_token

            
        
class Discriminator(nn.Module):
    """A CNN for text classification num_filters (int): This is the
    output dim for each convolutional layer, which is the number of
    "filters" learned by that layer.

    """
    def __init__(self, seq_len, num_classes, vocab_size, dis_emb_dim, 
                    filter_sizes, num_filters, start_token, goal_out_size, step_size, dropout_prob, l2_reg_lambda):
        super(Discriminator, self).__init__()
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.dis_emb_dim = dis_emb_dim
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.start_token = start_token
        self.goal_out_size = goal_out_size
        self.step_size = step_size
        self.dropout_prob = dropout_prob
        self.l2_reg_lambda = l2_reg_lambda
        self.num_filters_total = sum(self.num_filters)
        
        #Building up layers
        self.emb = nn.Embedding(self.vocab_size + 1, self.dis_emb_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_f, (f_size, self.dis_emb_dim)) for f_size, num_f in zip(self.filter_sizes, self.num_filters)
        ])
        self.highway = nn.Linear(self.num_filters_total, self.num_filters_total)
        #in_features = out_features = sum of num_festures
        self.dropout = nn.Dropout(p = self.dropout_prob)
        #Randomly zeroes some of the elements of the input tensor with probability p using Bernouli distribution
        #Each channel will be zeroed independently on every forward call
        self.fc = nn.Linear(self.num_filters_total, self.num_classes)
        
    def forward(self, x):
        """
        Argument:
            x: shape(batch_size * self.seq_len)
               type(Variable containing torch.LongTensor)
        Return:
            pred: shape(batch_size * 2)
                  For each sequence in the mini batch, output the probability
                  of it belonging to positive sample and negative sample.
            feature: shape(batch_size * self.num_filters_total)
                     Corresponding to f_t in original paper
            score: shape(batch_size, self.num_classes)
              
        """
        #1. Embedding Layer
        #2. Convolution + maxpool layer for each filter size
        #3. Combine all the pooled features into a prediction
        #4. Add highway
        #5. Add dropout. This is when feature should be extracted
        #6. Final unnormalized scores and predictions

        emb = self.emb(x).unsqueeze(1)
        convs = [F.relu(conv(emb)).squeeze(3) for conv in self.convs] # [batch_size * num_filter * seq_len]
        pooled_out = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs] # [batch_size * num_filter]
        pred = torch.cat(pooled_out, 1) # batch_size * sum(num_filters)
        #print("Pred size: {}".format(pred.size()))
        highway = self.highway(pred)
        #print("highway size: {}".format(highway.size()))
        highway = torch.sigmoid(highway)* F.relu(highway) + (1.0 - torch.sigmoid(highway))*pred
        features = self.dropout(highway)
        score = self.fc(features)
        pred = F.log_softmax(score, dim=1) #batch * num_classes
        return {"pred":pred, "feature":features, "score": score}

    def null_feature():
        return torch.zeros(self.dis_emb_dim)
    
    def l2_loss(self):
        W = self.fc.weight
        b = self.fc.bias
        l2_loss = torch.sum(W*W) + torch.sum(b*b)
        l2_loss = self.l2_reg_lambda * l2_loss
        return l2_loss
