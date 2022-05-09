#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import copy 

#### KL
def KL_divergence(teacher_batch_input, student_batch_input, reduction='sum'):
    """
    Compute the KL divergence of 2 batches of layers
    Args:
        teacher_batch_input: Size N x d
        student_batch_input: Size N x c
    
    Method: Kernel Density Estimation (KDE)
    Kernel: Gaussian
    Author: Nguyen Nang Hung
    """
    batch_student, _ = student_batch_input.shape
    batch_teacher, _ = teacher_batch_input.shape
    
    assert batch_teacher == batch_student, "Unmatched batch size"
    
    teacher_batch_input = teacher_batch_input.unsqueeze(1)
    student_batch_input = student_batch_input.unsqueeze(1)
    
    sub_s = student_batch_input - student_batch_input.transpose(0,1)
    sub_s_norm = torch.norm(sub_s, dim=2)
    sub_s_norm = sub_s_norm[sub_s_norm!=0].view(batch_student,-1)
    std_s = torch.std(sub_s_norm)
    mean_s = torch.mean(sub_s_norm)
    kernel_mtx_s = torch.pow(sub_s_norm - mean_s, 2) / (torch.pow(std_s, 2) + 0.001)
    kernel_mtx_s = torch.exp(-1/2 * kernel_mtx_s)
    kernel_mtx_s = kernel_mtx_s/torch.sum(kernel_mtx_s, dim=1, keepdim=True)
    
    sub_t = teacher_batch_input - teacher_batch_input.transpose(0,1)
    sub_t_norm = torch.norm(sub_t, dim=2)
    sub_t_norm = sub_t_norm[sub_t_norm!=0].view(batch_teacher,-1)
    std_t = torch.std(sub_t_norm)
    mean_t = torch.mean(sub_t_norm)
    kernel_mtx_t = torch.pow(sub_t_norm - mean_t, 2) / (torch.pow(std_t, 2) + 0.001)
    kernel_mtx_t = torch.exp(-1/2 * kernel_mtx_t)
    kernel_mtx_t = kernel_mtx_t/torch.sum(kernel_mtx_t, dim=1, keepdim=True)
    if reduction =='sum':
        kl = torch.sum(kernel_mtx_t * torch.log(kernel_mtx_t/kernel_mtx_s))
    elif reduction=='mean':
        kl = torch.mean(kernel_mtx_t * torch.log(kernel_mtx_t/kernel_mtx_s))

    return kl

#### FedProx 
def difference_models_norm_2(params1, params2):
    """Return the norm 2 difference between the two model parameters
    """
    
    norm=sum([torch.sum((params1[i]-params2[i])**2) 
        for i in range(len(params1))])
    
    return norm

#### Uncertainty
def relu_evidence(logits):
    return F.relu(logits)

def KL(alpha,K=10):
    beta=torch.ones(1,K)
    S_alpha = torch.sum(alpha,dim=1,keepdim=True)
    S_beta = torch.sum(beta,dim=1,keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha),dim=1,keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta),dim=1,keepdim=True) - torch.lgamma(S_beta)
    
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    
    kl = torch.sum((alpha - beta)*(dg1-dg0),dim=1,keepdim=True) + lnB + lnB_uni
    return kl

def mse_loss(labels, alpha, K, global_step, annealing_step): 
    p = F.one_hot(labels, num_classes=K)

    S = torch.sum(alpha, dim=1, keepdim=True) 
    E = alpha - 1
    m = alpha / S
    
    A = torch.sum((p-m)**2, dim=1, keepdim=True) 
    B = torch.sum(alpha*(S-alpha)/(S*S*(S+1)), axis=1, keepdim=True) 
    
    annealing_coef = min(1.0,global_step/annealing_step)
    
    alp = E*(1-p) + 1 
    C =  annealing_coef * KL(alp,K)
    return (A + B) + C

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Default criterion set to NLL loss function
        # self.criterion = nn.NLLLoss().to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)/10), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test)/10), shuffle=False)
        return trainloader, validloader, testloader

    def update_weights(self, model, global_round, mu=0, uncertainty=False):
        global_params = copy.deepcopy(list(model.parameters()))
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                _, logits = model(images)
                if uncertainty:
                    evidence = relu_evidence(logits)
                    alpha = evidence + 1
                    K=logits.shape[-1]
                    u = K / torch.sum(alpha, dim=1, keepdim=True) #uncertainty
                    prob = alpha/torch.sum(alpha, 1, keepdim=True) 
                    
                    loss = torch.mean(mse_loss(labels, alpha, K, iter*len(self.trainloader)+batch_idx, 10*len(self.trainloader)))

                else:                    
                    loss = self.criterion(logits, labels)
                
                loss_ = loss + mu *difference_models_norm_2(list(model.parameters()), global_params)
                loss_.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def update_weights_kd(self, global_model, global_round, T=1, alpha=0.2, kl='kl', reduction='sum'):
        user_model = copy.deepcopy(global_model)
        global_model.eval()
        user_model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(user_model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(user_model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                user_model.zero_grad()
                embs, logits = user_model(images)
                loss1 = self.criterion(logits, labels)
                global_embs, global_logits = global_model(images)
                
                if kl=='kl_comb':
                    loss2 = KL_divergence(global_embs, embs, reduction) + nn.KLDivLoss(reduction=reduction)(F.log_softmax(logits/T, dim=1),
                             F.softmax(global_logits/T, dim=1)) * (T * T)
                elif kl=='kl_dist':
                    loss2 = KL_divergence(global_embs, embs, reduction)
                else:
                    loss2 = nn.KLDivLoss(reduction=reduction)(F.log_softmax(logits/T, dim=1),
                             F.softmax(global_logits/T, dim=1)) * (T * T)

                loss = loss1 + alpha*loss2
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss1.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return user_model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model, uncertainty):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        us = []
        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            _,  outputs = model(images)
            if uncertainty:
                evidence = relu_evidence(outputs)
                alpha = evidence + 1
                K=outputs.shape[-1]
                u = K / torch.sum(alpha, dim=1, keepdim=True) #uncertainty
                us += u.reshape((-1,)).detach().cpu().numpy().tolist()
            else:
                batch_loss = self.criterion(outputs, labels)
                loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total

        if uncertainty:
            return accuracy, sum(us)/len(us)
        else:
            return accuracy, loss


def test_inference(args, model, test_dataset, uncertainty):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    us = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        _, outputs = model(images)
        if uncertainty:
            evidence = relu_evidence(outputs)
            alpha = evidence + 1
            K=outputs.shape[-1]
            u = K / torch.sum(alpha, dim=1, keepdim=True) #uncertainty
            us += u.reshape((-1,)).detach().cpu().numpy().tolist()
        else:
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    if uncertainty:
        return accuracy, sum(us)/len(us)
    else:
        return accuracy, loss
