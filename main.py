import argparse
import pickle as pkl
import numpy as np
import json 
import glob
import os #for checkpoint management
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_

from data_iter import real_data_loader, dis_data_loader, dis_l2_data_loader, vector_data_loader
from utils import recurrent_func, loss_func, get_sample, get_rewards, getNullFeature
from Discriminator import Discriminator, Discriminator_l2
from Generator import Generator, Generator_l2
from target_lstm import TargetLSTM

###################################################################
# Debug Node
#############################################

# We have a global variable here which we check against elsewhere in
# the code. If it's true, then only ever perform one run of
# everything. That is, each training step has only 1 epoch, a batch
# size of 1, and 1 batch to process.

# We do it this way because my machine is not capable of handling the
# demands of a full set of training, and so we really only need to
# check for exceptions and other testing purposes. You know, make sure
# the code isn't going hit some bug while processing using the Azure
# Compute Cluster.

DEBUG_NODE = True

##################################################################

#Files
POSITIVE_FILE = "real.data"
NEGATIVE_FILE = "gene.data"

def get_params(filePath):
    with open(filePath, 'r') as f:
        params = json.load(f)
    f.close()
    return params

def get_arguments():
    train_params = get_params("./params/train_params.json")
    leak_gan_params = get_params("./params/leak_gan_params.json")
    target_params = get_params("./params/target_params.json")
    dis_data_params = get_params("./params/dis_data_params.json")
    real_data_params = get_params("./params/real_data_params.json")
    return {
        "train_params": train_params,
        "leak_gan_params": leak_gan_params,
        "target_params": target_params,
        "dis_data_params": dis_data_params,
        "real_data_params" : real_data_params
    }

def get_arguments_l2():
    train_params = get_params("./params/train_l2_params.json")
    leak_gan_params = get_params("./params/leak_gan_l2_params.json")
    target_params = get_params("./params/target_l2_params.json")
    dis_data_params = get_params("./params/dis_l2_data_params.json")
    real_data_params = get_params("./params/real_l2_data_params.json")
    return {
        "train_params": train_params,
        "leak_gan_params": leak_gan_params,
        "target_params": target_params,
        "dis_data_params": dis_data_params,
        "real_data_params" : real_data_params
    }

#List of models
def prepare_model_dict(filename, lvl, use_cuda=False):
    #############################################################################
    #

    print("Preparing Dictionary of Models. Discriminator, Generator, Optimizer " \
          "Scheduler")

    ##############################################################################
    
    f = open(filename)
    params = json.load(f)
    f.close()
    discriminator_params = params["discriminator_params"]
    generator_params = params["generator_params"]
    worker_params = generator_params["worker_params"]
    manager_params = generator_params["manager_params"]
    discriminator_params["goal_out_size"] = sum(
        discriminator_params["num_filters"]
    )
    worker_params["goal_out_size"] = discriminator_params["goal_out_size"]
    manager_params["goal_out_size"] = discriminator_params["goal_out_size"]

    if lvl == "l1":
        discriminator = Discriminator(**discriminator_params)
        generator = Generator(worker_params, manager_params,
                              generator_params["step_size"])
    elif lvl == "l2":
        discriminator = Discriminator_l2(**discriminator_params)
        generator = Generator_l2(worker_params, manager_params, generator_params["step_size"])
        
    if use_cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
    model_dict = {"generator": generator, "discriminator": discriminator}
    return model_dict

#List of optimizers
def prepare_optimizer_dict(model_dict, lr_dict): #lr_dict = learning rate 
    generator = model_dict["generator"]
    discriminator = model_dict["discriminator"]
    worker = generator.worker
    manager = generator.manager

    m_lr = lr_dict["manager"]
    w_lr = lr_dict["worker"]
    d_lr = lr_dict["discriminator"]

    w_optimizer = optim.Adam(worker.parameters(), lr=w_lr)
    m_optimizer = optim.Adam(manager.parameters(), lr=m_lr)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=d_lr)

    return {"worker": w_optimizer, "manager": m_optimizer,
            "discriminator": d_optimizer}

#List of Learning rate Schedulers
def prepare_scheduler_dict(optmizer_dict, step_size=200, gamma=0.99):
    w_optimizer = optmizer_dict["worker"]
    m_optimizer = optmizer_dict["manager"]
    d_optimizer = optmizer_dict["discriminator"]

    w_scheduler = optim.lr_scheduler.StepLR(w_optimizer, step_size=step_size,
                                            gamma=gamma)
    m_scheduler = optim.lr_scheduler.StepLR(m_optimizer, step_size=step_size,
                                            gamma=gamma)
    d_scheduler = optim.lr_scheduler.StepLR(d_optimizer, step_size=step_size,
                                            gamma=gamma)
    return {"worker": w_scheduler, "manager": m_scheduler,
            "discriminator": d_scheduler}

#Pretraining the Generator
def pretrain_generator(model_dict, optimizer_dict, scheduler_dict, dataloader, vocab_size, max_norm=5.0, use_cuda=False, epoch=1, tot_epochs=100):
    #get the models of generator
    generator = model_dict["generator"]
    worker = generator.worker
    manager = generator.manager
    #get the optimizers
    m_optimizer = optimizer_dict["manager"]
    w_optimizer = optimizer_dict["worker"]

    # Testing to make sure python environment stops bitching.
    m_optimizer.step()
    w_optimizer.step()
    
    m_optimizer.zero_grad()
    w_optimizer.zero_grad()

    m_lr_scheduler = scheduler_dict["manager"]
    w_lr_scheduler = scheduler_dict["worker"]
    """
     Perform pretrain step for real data
    """
    
    for i, sample in enumerate(dataloader):
        #print("DataLoader: {}".format(dataloader))
        m_lr_scheduler.step()
        w_lr_scheduler.step()


        print("Current Sample: %s at index %d" % (sample, i))
          
        sample = Variable(sample)
        if use_cuda:
            sample = sample.cuda(non_blocking=True)
        
        # Calculate pretrain loss
        if (sample.size() == torch.zeros([64, 20]).size()): #sometimes smaller than 64 (16) is passed, so this if statement disables it
            #print("Sample size: {}".format(sample.size()))
            pre_rets = recurrent_func("pre")(model_dict, sample, use_cuda)
            real_goal = pre_rets["real_goal"]
            prediction = pre_rets["prediction"]
            delta_feature = pre_rets["delta_feature"]

            m_loss = loss_func("pre_manager")(real_goal, delta_feature)
            torch.autograd.grad(m_loss, manager.parameters())
            clip_grad_norm_(manager.parameters(), max_norm=max_norm)
            m_optimizer.step()
            m_optimizer.zero_grad()
            
            w_loss = loss_func("pre_worker")(sample, prediction, vocab_size, use_cuda)
            torch.autograd.grad(w_loss, worker.parameters())
            clip_grad_norm_(worker.parameters(), max_norm=max_norm)
            w_optimizer.step()
            w_optimizer.zero_grad()
          
            if i == 63:
                print("Pre-Manager Loss: {:.5f}, Pre-Worker Loss: {:.5f}\n".format(m_loss, w_loss))
          
            if DEBUG_NODE == True:
                print("DEBUG_NODE flag detected! Ending pretrain step early. Wrapping Up.")
                break
    
    """
    Update model_dict, optimizer_dict, and scheduler_dict
    """

    print("Updating and storing changes to models.")
    
    generator.worker = worker
    generator.manager = manager
    model_dict["generator"] = generator

    optimizer_dict["manager"] = m_optimizer
    optimizer_dict["worker"] = w_optimizer

    scheduler_dict["manager"] = m_lr_scheduler
    scheduler_dict["worker"] = w_lr_scheduler

    return model_dict, optimizer_dict, scheduler_dict

def pretrain_generator_l2(model_dict_l2, optimizer_dict, scheduler_dict):
    generator_l2 = model_dict_l2["generator_l2"]
    worker = generator_l2.worker
    manager = generator_l2.manager

    # Again, optimizers and scheduler's are ignored at this
    # point. We're just retrofitting the flow of the control logic for
    # now, thank you very much. ;^D

    # Just an empty dictionary for now, so python will shut up.
    real_l2_data_params = get_params("./params/real_l2_data_params.json")
    
    if use_cuda:
        real_l2_data_params["pin_memory"] = True

    dataloader  = real_data_loader(**real_l2_data_params)

    # optimizer and scheduler stuff should go here.
    
    for i, sample in enumerate(dataloader):
        # Again, just retrofitting logic flow. Anything I write
        # here will most definitely raise an exception. Still,
        # it's good to know that this is where we'll be feeding
        # the data into the level 2 generator.

        # Now remember! The level 2 generator is fed FEATURAL
        # VECTORS, not the actual corpus text itself. This is very
        # important, because IT WILL AFFECT THE INITIALIZATION
        # PARAMETERS OF THE L2 AND L3 GENERATORS/DISCRIMINATORS!!!

        continue

    # Save the changed models back in their respective slots.
    generator_l2.worker = worker
    generator_l2.manager = manager
    model_dict_l2["generator_l2"] = generator_l2

    # And fuck the optimizers and schedulers until I know what they're
    # good for.

    return model_dict_l2, False, False

def generate_samples(model_dict, negative_file, batch_size,
                     use_cuda=False, temperature=1.0):
    print("Generating Samples")
    
    neg_data = []
    for i in range(batch_size):
        sample = get_sample(model_dict, use_cuda, temperature)

        if i < 25:
            print("Generated: %s" % sample)
        elif i == 25:
            print("Omitting remaining samples for brevity.")
        
        sample = sample.cpu()
        neg_data.append(sample.data.numpy())
    neg_data = np.concatenate(neg_data, axis=0)
    np.save(negative_file, neg_data)

def pretrain_discriminator(model_dict, optimizer_dict, scheduler_dict,
                           dis_dataloader_params, vocab_size, positive_file,
                           negative_file, batch_size, epochs, use_cuda=False, temperature=1.0):
    discriminator = model_dict["discriminator"]

    d_optimizer = optimizer_dict["discriminator"]
    d_lr_scheduler = scheduler_dict["discriminator"]

    generate_samples(model_dict, negative_file, batch_size, use_cuda, temperature)
    dataloader = dis_data_loader(**dis_dataloader_params) #this is where data iterator is used

    cross_entropy = nn.CrossEntropyLoss() #this one is similar to NLL (negative log likelihood)
    if use_cuda:
        cross_entropy = cross_entropy.cuda()

    # Added 2 variables
    features  = "undefined"
    cpuTensor = None
    
    for epoch in range(epochs):
        for i, sample in enumerate(dataloader):
            d_optimizer.zero_grad()
            data, label = sample["data"], sample["label"] #initialize sample variables
            data = Variable(data)
            label = Variable(label)
            if use_cuda:
                data = data.cuda()
                label = label.cuda()
            outs = discriminator(data)

            ##################
            # Injection Point: Collect featural data in numpy array.


            # NOTE: We need to check which elements of the batch_size
            # that the discriminator predicts to be as part of the
            # positive sample, and only extract the featural data of
            # those elements to the featural corpus file. We also need
            # to reshape the cpuTensor elements so that each element
            # along axis-0 corresponds to the featural embedding
            # vector for only one input sentence.
            
            print("Found data: %s. \n\n Featural Output: %s , %s"
                  % (sample["data"], outs["feature"], outs["feature"].tolist()))

            # First, make sure that we haven't already initialized the
            # numpy container.

            cpuTensor = outs["feature"].cpu()
            
            if features  == "undefined":
                # If here, then we first need to create a numpy array
                # with the same shape as the featural output from the
                # discriminator.
                #features = np.empty(cpuTensor.detach().numpy().shape)
                features = []

            # Now we're guaranteed that we can append the featural
            # output to the numpy array.
            #features = np.append(features, cpuTensor.detach().numpy(), axis=0)
            features.append(cpuTensor.detach().numpy().tolist())

            #
            #################
            
            loss = cross_entropy(outs["score"], label.view(-1)) + discriminator.l2_loss()
            d_lr_scheduler.step()
            loss.backward()
            d_optimizer.step()
            if i == 9:
                print("Pre-Discriminator loss: {:.5f}".format(loss))

            if DEBUG_NODE == True:
                break
        if DEBUG_NODE == True:
            break

    ################
    # Injection Point: Save collected featural data to 'featural_vectors.npy'

    with open('./data/featural_vectors.npy', 'wb') as fd:
        features = np.array(features)
        np.save(fd, features)

    #
    ################
    
    model_dict["discriminator"] = discriminator
    optimizer_dict["discriminator"] = d_optimizer
    scheduler_dict["discriminator"] = d_lr_scheduler
    return model_dict, optimizer_dict, scheduler_dict

def pretrain_discriminator_l2(model_dict_l2, optimizer_dict_l2, scheduler_dict_l2, dis_l2_data_params, use_cuda=False, temperature=1.0):
    """
    Pretrain the 2nd level discriminator.
    """
    discriminator_l2 = model_dict_l2["discriminator_l2"]
    batch_size = dis_l2_data_params["batch_size"]

    d_optimizer = optimizer_dict_l2["discriminator"]
    d_lr_scheduler = scheduler_dict_l2["discriminator"]

    neg_l2_fd = dis_l2_data_params["negative_filepath"]
    
    generate_samples(model_dict_l2, neg_l2_fd, batch_size, use_cuda, temperature)
    dataloader = dis_data_loader(**dis_l2_data_params)

    # I don't yet know what cross_entropy is, but it was in the other
    # method, so it's in here, as well.
    cross_entropy = nn.CrossEntropyLoss()
    if use_cuda:
        cross_entropy = cross_entropy.cuda()

    epochs = 1
    for epoch in range(epochs):
        for i, sample in enumerate(dataloader):
            d_optimizer.zero_grad()
            data, label = sample["data"], sample["label"]

            data = Variable(data)
            label = Variable(label)
            
            if use_cuda:
                data = data.cuda()
                lbel = label.cuda()

            outs = discriminator_l2(data)

            ## SITE: FUTURE INJECTION POINT.

            loss = cross_entropy(outs["score"], label.view(-1)) + discriminator_l2.l2_loss()
            d_lr_schedular.step()
            loss.backward()
            d_optimizer.step()

            if i == i:
                print("Pre-Discrimination loss: {:.5f}".format(loss))

            if DEBUG_NODE == True:
                break
        if DEBUG_NODE == True:
            break
                
    # After training we need to resave the changes to our
    # discriminator and generator. The optimizer and scheduler
    # dictionaries as well, but since I don't know what those are, I'm
    # not going to write it here yet.
            
    model_dict_l2["discriminator_l2"] = discriminator_l2
    optimizer_dict_l2["discriminator"] = d_optimizer
    scheduler_dict_l2["discriminator"] = d_lr_schedular

    return model_dict_l2, optimizer_dict_l2, scheduler_dict_l2

def adv_l2_train(model_dict, model_dict_l2, optimizer_dict, scheduler_dict, use_cuda=False, temperature=1.0, epoch=1, total_epoch=1):
    pass

#Adversarial training 
def adversarial_train(model_dict, optimizer_dict, scheduler_dict, dis_dataloader_params,
                      vocab_size, pos_file, neg_file, batch_size, gen_train_num=1,
                      dis_train_epoch=5, dis_train_num=3, max_norm=5.0,
                      rollout_num=4, use_cuda=False, temperature=1.0, epoch=1, tot_epoch=100):
    """
        Get all the models, optimizer and schedulers
    """                     
    generator = model_dict["generator"]
    discriminator = model_dict ["discriminator"]
    worker = generator.worker
    manager = generator.manager

    m_optimizer = optimizer_dict["manager"]
    w_optimizer = optimizer_dict["worker"]
    d_optimizer = optimizer_dict["discriminator"]

    #Why zero grad only m and w?
    m_optimizer.zero_grad()
    w_optimizer.zero_grad()

    m_lr_scheduler = scheduler_dict["manager"]
    w_lr_scheduler = scheduler_dict["worker"]
    d_lr_scheduler = scheduler_dict["discriminator"]

    #Adversarial training for generator
    for _ in range(gen_train_num):
        m_lr_scheduler.step()
        w_lr_scheduler.step()

        m_optimizer.zero_grad()
        w_optimizer.zero_grad()

        #get all the return values
        adv_rets = recurrent_func("adv")(model_dict, use_cuda)
        real_goal = adv_rets["real_goal"]
        all_goal = adv_rets["all_goal"]
        prediction = adv_rets["prediction"]
        delta_feature = adv_rets["delta_feature"]
        delta_feature_for_worker = adv_rets["delta_feature_for_worker"]
        gen_token = adv_rets["gen_token"]

        rewards = get_rewards(model_dict, gen_token, rollout_num, use_cuda)
        m_loss = loss_func("adv_manager")(rewards, real_goal, delta_feature)
        w_loss = loss_func("adv_worker")(all_goal, delta_feature_for_worker, gen_token, prediction, vocab_size, use_cuda)

        torch.autograd.grad(m_loss, manager.parameters()) #based on loss improve the parameters
        torch.autograd.grad(w_loss, worker.parameters())
        clip_grad_norm_(manager.parameters(), max_norm)
        clip_grad_norm_(worker.parameters(), max_norm)
        m_optimizer.step()
        w_optimizer.step()
        print("Adv-Manager loss: {:.5f} Adv-Worker loss: {:.5f}".format(m_loss, w_loss))

        if DEBUG_NODE == True:
            break
    
    del adv_rets
    del real_goal
    del all_goal
    del prediction
    del delta_feature
    del delta_feature_for_worker
    del gen_token
    del rewards

    #Adversarial training for discriminator
    for n in range(dis_train_epoch):
        generate_samples(model_dict, neg_file, batch_size, use_cuda, temperature)
        dis_dataloader_params["positive_filepath"] = pos_file
        dis_dataloader_params["negative_filepath"] = neg_file
        dataloader = dis_data_loader(**dis_dataloader_params)

        cross_entropy = nn.CrossEntropyLoss()
        if use_cuda:
            cross_entropy = cross_entropy.cuda()
        """
        for d-steps do
            Use current G, θm,θw to generate negative examples and combine with given positive examples S 
            Train discriminator Dφ for k epochs by Eq. (2)
        end for
        """
        for _ in range(dis_train_num): 
            for i, sample in enumerate(dataloader):
                data, label = sample["data"], sample["label"]
                data = Variable(data)
                label = Variable(label)
                if use_cuda:
                    data = data.cuda(non_blocking=True)
                    label = label.cuda(non_blocking=True)
                outs = discriminator(data)
                loss = cross_entropy(outs["score"], label.view(-1)) + discriminator.l2_loss()
                d_optimizer.zero_grad()
                d_lr_scheduler.step()
                loss.backward()
                d_optimizer.step()

                if DEBUG_NODE == True:
                    break
            if DEBUG_NODE == True:
                break
                
        print("{}/{} Adv-Discriminator Loss: {:.5f}".format(n, range(dis_train_epoch),loss))
    #Save all changes
    model_dict["discriminator"] = discriminator
    generator.worker = worker
    generator.manager = manager
    model_dict["generator"] = generator

    optimizer_dict["manager"] = m_optimizer
    optimizer_dict["worker"] = w_optimizer
    optimizer_dict["discriminator"] = d_optimizer

    scheduler_dict["manager"] = m_lr_scheduler
    scheduler_dict["worker"] = w_lr_scheduler
    scheduler_dict["disciminator"] = d_lr_scheduler

    return model_dict, optimizer_dict, scheduler_dict


# def save_checkpoint(model_dict, optimizer_dict, scheduler_dict, ckpt_num, replace=False):
#     file_name = "checkpoint" + str(ckpt_num) + ".pth.tar"
#     torch.save({"model_dict": model_dict, "optimizer_dict": optimizer_dict, "scheduler_dict": scheduler_dict, "ckpt_num": ckpt_num}, file_name)
#     if replace:
#         ckpts = glob.glob("checkpoint*")
#         ckpt_nums = [int(x.split('.')[0][10:]) for x in ckpts]
#         oldest_ckpt = "checkpoint" + str(min(ckpt_nums)) + ".pth.tar"
#         os.remove(oldest_ckpt)

# def restore_checkpoint(ckpt_path):
#     checkpoint = torch.load(ckpt_path)
#     return checkpoint

def dis_l2_data_params(model_dict, **kwargs):
    par_data_fd = open(kwargs["par_data_filepath"], 'rb')
    pos_data_fd = open(kwargs["positive_filepath"], 'w+b')
    max_parlength = kwargs["paragraph_length"]

    vecdata = numpy.load(pos_data_fd)
    pardata = numpy.load(par_data_fd)
    
    paddata = []
    current_index = 0
    
    for parlength in pardata:
        for par in range(max_parlength):
            if parlength > max_parlength or parlength <= 0:
                continue
            else:
                if par <= parlength:
                    paddata.append(vecdata[(current_index+(par+1))])
                else:
                    paddata.append(model_dict["discriminator"].getNullFeature(model_dict, use_cuda))
        
        if parlength <= 0:
            continue
        else:
            current_index += parlength

    numpy.save(pos_data_fd, paddata)

def fetch_l2_corpus(model_dict, positive_filepath, dis_l2_data_params, use_cuda):
    discriminator = model_dict["discriminator"]
    
    dataloader = vector_data_loader(positive_filepath, batch_size=1, shuffle=False,
                                    num_workers=4, pin_memory=dis_l2_data_params["pin_memory"])

    vectors = []

    sout = open("./data/sout.txt", 'w')
    
    for i, sample in enumerate(dataloader):
        data, label = sample["data"], sample["label"]
        data = Variable(data)
        label = Variable(label)

        if use_cuda:
            data = data.cuda()
            label = label.cuda()

        outs = discriminator(data)
        print("Featural Vector: %s" % outs["feature"].tolist())
        print("size of vector: %s" % (list(outs["feature"].size())), file=sout)
        featural_vector = outs["feature"].tolist()
        vectors.append(featural_vector)

    sout.close()
    featural_data = np.array(vectors)

    # Remove the 'batch' axis.
    axis0 = featural_data.shape[0]
    print("axis0: %s " % axis0)
    axis1 = featural_data.shape[2]
    print("axis1: %s " % axis1)
    
    featural_data = featural_data.reshape((axis0, axis1))
    print("Featural Data Initial Shape: %s" % list(featural_data.shape))
    return featural_data
    
# This method should open up the positive_filepath file, and using
# other parameters, pad all the data inside and generally make it so
# that the data is the correct shape for being fed into the l2
# discriminator.
    
def dis_l2_pad_pars(vectors, dis_l2_data_params, use_cuda=False, temperature=1.0):
    par_filepath = dis_l2_data_params["par_data_filepath"]
    par_len = dis_l2_data_params["paragraph_length"]

    # par_fd = open(par_filepath, 'rb')
    par_data = np.load(par_filepath)

    # NOTE: It is *PRECISELY* at this location that we will need to
    # ascertain whether the shape of our positive data is apropriate,
    # and if necessary, modify the shape before proceeding.

    # Though at the current moment, the code is flying and there's
    # other stuff that needs to be taken care of, so we proceed for
    # now as if the shape of the positive data is correct.

    flag_p = False
    tmpPrimary = "undefined"
    pnum = 0

    for i in range(len(par_data)):
        if (pnum + par_data[i]) >= len(vectors):
            continue

        if par_data[i] > par_len:
            pnum += par_data[i]
            continue

        if par_data[i] <= par_len:
            tmpSecondary = np.array([vectors[pnum:(pnum+par_data[i])]])            
            for j in range(par_len - par_data[i]):
                tmpSecondary = np.append(tmpSecondary, [[np.zeros(len(vectors[0]))]], 1)

            if flag_p == False:
                tmpPrimary = np.array(tmpSecondary)
                flag_p = True
            else:
                tmpPrimary = np.append(tmpPrimary, tmpSecondary, 0)

            pnum += par_data[i]

    return tmpPrimary
    
    
def main():
    """
    Get all parameters
    """

    #############################################################################
    #
    # Basic Preamble
    
    print("#########################################################################")
    print("# Welcome to ReLeakGan version 0.1a! This is a recursive extension of   #\n" \
          "# LeakGAN machine learning system for stochastic long text generation   #\n" \
          "#                                                                       #\n" \
          "# Please pardon our dust, as we are currently retrofitting the original #\n" \
          "# LeakGAN implementation control and logic flow. As a result, the       #\n" \
          "# current version (this one!) should not be expected to return usable   #\n" \
          "# results.                                                              #\n" \
          "#########################################################################\n")

    # user_input = 'x'
    # yes_response = 'y'
    # no_response = 'n'

    # while user_input != yes_response and user_input != no_response:
    #     print("Proceed? (y/n)")
    #     user_input = str(input())

    #     if user_input == yes_response:
    #         break
    #     elif user_input == no_response:
    #         return

    #############################################################################
    
    param_dict = get_arguments()
    # Disable cuda for now. We will re-enable when I submit to Azure
    #Compute Cluster. Note: Technically we may still use cuda
    #locally. Weird.
    use_cuda = torch.cuda.is_available()
    # use_cuda = False
    
    #Random seed
    torch.manual_seed(param_dict["train_params"]["seed"])
    #Pretrain step
    #checkpoint_path = param_dict["train_params"]["checkpoint_path"]
    # if checkpoint_path is not None:
    #     checkpoint = restore_checkpoint(checkpoint_path)
    #     model_dict = checkpoint["model_dict"]
    #     optimizer_dict = checkpoint["optimizer_dict"]
    #     scheduler_dict = checkpoint["scheduler_dict"]
    #     ckpt_num = checkpoint["ckpt_num"]
    # else:
    
    model_dict = prepare_model_dict("./params/leak_gan_params.json", "l1", use_cuda)
    lr_dict = param_dict["train_params"]["lr_dict"]
    optimizer_dict = prepare_optimizer_dict(model_dict, lr_dict)
    gamma = param_dict["train_params"]["decay_rate"]
    step_size = param_dict["train_params"]["decay_step_size"]
    scheduler_dict = prepare_scheduler_dict(optimizer_dict, gamma=gamma, step_size=step_size)
    
    #Pretrain discriminator
    print ("#########################################################################")
    print ("Start Pretraining Discriminator...")
    with open("./params/dis_data_params.json", 'r') as f:
        dis_data_params = json.load(f)
    if use_cuda:
        dis_data_params["pin_memory"] = True
    f.close()
    pos_file = dis_data_params["positive_filepath"]
    neg_file = dis_data_params["negative_filepath"]
    batch_size = param_dict["train_params"]["generated_num"]
    vocab_size = param_dict["leak_gan_params"]["discriminator_params"]["vocab_size"]
    for i in range(param_dict["train_params"]["pre_dis_epoch_num"]):
        print("Epoch: {}/{}  Pre-Discriminator".format(i, param_dict["train_params"]["pre_dis_epoch_num"]))
        model_dict, optimizer_dict, scheduler_dict = pretrain_discriminator(model_dict, optimizer_dict, scheduler_dict, dis_data_params, vocab_size=vocab_size, positive_file=pos_file, negative_file=neg_file, batch_size=batch_size, epochs=1, use_cuda=use_cuda)
    # ckpt_num = 0
    # save_checkpoint(model_dict, optimizer_dict, scheduler_dict, ckpt_num)

    #Pretrain generator 
    print ("#########################################################################")
    print ("Start Pretraining Generator...")
    real_data_params = param_dict["real_data_params"]
    if use_cuda:
        real_data_params["pin_memory"] = True
    r_dataloader = real_data_loader(**real_data_params)
    for epoch in range(param_dict["train_params"]["pre_gen_epoch_num"]):
        print("Epoch: {}/{}  Pre-Generator".format(epoch, param_dict["train_params"]["pre_gen_epoch_num"]))
        model_dict, optimizer_dict, scheduler_dict = pretrain_generator(model_dict, optimizer_dict, scheduler_dict, r_dataloader, vocab_size=vocab_size, use_cuda=use_cuda, epoch=epoch, tot_epochs=range(param_dict["train_params"]["pre_gen_epoch_num"]))
    #Finish pretrain and save the checkpoint
    #save_checkpoint(model_dict, optimizer_dict, scheduler_dict, ckpt_num)
    
    
    #ckpt_num = 1
    #Adversarial train of D and G
    print ("#########################################################################")
    print ("Start Adversarial Training...")
    vocab_size = param_dict["leak_gan_params"]["discriminator_params"]["vocab_size"]
    save_num = param_dict["train_params"]["save_num"] #save checkpoint after this number of repetitions
    replace_num = param_dict["train_params"]["replace_num"]

    for epoch in range(param_dict["train_params"]["total_epoch"]):
        print("Epoch: {}/{}  Adv".format(epoch, param_dict["train_params"]["total_epoch"]))
        model_dict, optimizer_dict, scheduler_dict = adversarial_train(model_dict, optimizer_dict, scheduler_dict, dis_data_params, vocab_size=vocab_size, pos_file=pos_file, neg_file=neg_file, batch_size=batch_size, use_cuda=use_cuda, epoch=epoch, tot_epoch=param_dict["train_params"]["total_epoch"])

    ##############################################################################
    #
    # Epilogue Section

        
    print("#########################################################################")
    print("# After this point, the control flow and logic of the application is    #\n" \
          "# unstable, and the program will most certainly crash near immediately  #\n" \
          "# after this point. Still, feel free to continue if you want to see.    #\n" \
          "# ;^D                                                                   #\n" \
          "#########################################################################\n")

    # user_input = 'x'
    # yes_response = 'y'
    # no_response = 'n'

    # while user_input != yes_response and user_input != no_response:
    #     print("Proceed? (y/n)")
    #     user_input = str(input())

    #     if user_input == yes_response:
    #         break
    #     elif user_input == no_response:
    #         printf("Thank you for using ReLeakGAN version 0.1a!")
    #         return
    #
    #############################################################################

    # Prepare model dictionary, and fetch level 2 parameters.
    param_dict_l2 = get_arguments_l2()

    if use_cuda:
        param_dict_l2["dis_data_params"]["pin_memory"] = True
    
    model_dict_l2 = prepare_model_dict("./params/leak_gan_l2_params.json", "l2", use_cuda)
    optimizer_dict_l2 = prepare_optimizer_dict(model_dict_l2, lr_dict)
    scheduler_dict_l2 = prepare_scheduler_dict(optimizer_dict_l2, gamma=gamma, step_size=step_size)

    # We need to re-run the original training corpus through the l1
    # discriminator, and have it spit out featural vectors for us.

    l2_corpus = fetch_l2_corpus(model_dict, param_dict["dis_data_params"]["positive_filepath"],
                                param_dict_l2["dis_data_params"], use_cuda=use_cuda)

    sout = open("./data/sout.txt", 'w')
    print("Shape of l2_corpus: %s" % (list(l2_corpus.shape)), file=sout)

    # We need to group the vectors into paragraphs.
    l2_corpus = dis_l2_pad_pars(l2_corpus, param_dict_l2["dis_data_params"], use_cuda=use_cuda)
    np.save("./data/featural_vectors.npy", l2_corpus)

    print("Shape of l2_corpus after padding: %s" % (list(l2_corpus.shape)), file=sout)

    print("######################################################################")
    print("# Now Pretraining Level 2 Discriminator... please stand by.          #")
    print("######################################################################")
    
    with open("./params/dis_l2_data_params.json", 'r') as f:
        dis_l2_data_params = json.load(f)
    if use_cuda:
        dis_l2_data_params["pin_memory"] = True
    f.close()
    
    num_epochs = param_dict["train_params"]["pre_dis_l2_epoch_num"]
    
    for i in range(num_epochs):
        print("Epoch: {}/{} Pre-Discrimination".format(i, num_epochs))
        model_dict_l2, optimizer_dict_l2, scheduler_dict_l2 = pretrain_discriminator_l2(model_dict_l2,
                                                                                        optimizer_dict_l2,
                                                                                        scheduler_dict_l2,
                                                                                        dis_l2_data_params,
                                                                                        use_cuda=use_cuda)

        if DEBUG_NODE == True:
            break

    # Now we need to pretrain the l2 generator.

    print("######################################################################")
    print("# Now Pretraining Level 2 Generator... please stand by.              #")
    print("######################################################################")

    num_epochs = param_dict["train_params"]["pre_gen_l2_epoch_num"]
    
    for i in range(num_epochs):
        print("Epoch: {}/{} Pre-Generation".format(i, num_epochs))
        model_dict_l2, optimizer_dict_l2, scheduler_dict_l2 = pretrain_generator_l2(model_dict_l2, optimizer_dict, schedular_dict, use_cuda=use_cuda)

        if DEBUG_NODE == True:
            break

    # * * *
    
    # Next up, adversarial training. But we need to be careful,
    # because we are training BOTH the l1 AND the l2 discriminators
    # and generators in tandem.

    
    
    ##############################################################################
    # Please note: Comments below this point MAY NOT actually reflect
    # the location in the code where their subject matter is being
    # implemented.
    ##############################################################################
    
    # Receiving the sentence-corpus like this, we simply need to send
    # it to our wrapped object to convert the text sentence by
    # sentence into feature vectors, and then format and process these
    # feature vectors into the sentence level corpus. The wrapper
    # object should be able to supply the relevant numpy array via a
    # simple accessor function.

    # In continued main, we call the relevant accessor, which does all
    # of the preprocessing and returns the input corpus numpy array.

    # We get the numpy array, combine it with our second level
    # generator output, combine the two data sets, shuffle, and then
    # send the data to the 2nd level discriminator for pretraining.

    # Now, in true recursive fashion, the 2nd level discriminator needs
    # to know what data is real and fake. So we need to generate
    # featural vectors from the 2nd level generator, put them into
    # a negative dataset, concatenate with the positive dataset
    # and only then fed them to the discriminator.

    # The positive featural dataset can be used directly to train
    # the 2nd level generator.

    # We will need to implement an additional recurrent function for
    # the first level generator given 2nd level adversarial training.

    # Same thing for the 1st level discriminator, because during 2nd
    # level adversarial training, we need to have the data flow
    # through both sets of discriminators and generators.

    # Fortunately, I think I see how this can be done via utils.py

    # The mechanism for implementing the 3rd level is going to be
    # very similar, so I won't explicitly detail it here. Let me get
    # 2nd level up and running first. Then worry about the 3rd level.
               
if __name__ == "__main__":
    main()
