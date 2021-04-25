import torch
import numpy

def restore_checkpoint():
    checkpoint = torch.load("../checkpoints/checkpoint.pth.tar")
    return checkpoint

def fetch_l2_corpus(model_dict, positive_filepath, dis_l2_data_params, use_cuda):
    discriminator = model_dict["discriminator"]
    
    dataloader = vector_data_loader(positive_filepath, batch_size=1, shuffle=False,
                                    num_workers=4, pin_memory=dis_l2_data_params["pin_memory"])

    vectors = "undefined"
    flag_v = False
    
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

        featural_vector = outs["feature"]
        
        if flag_v == False:
            vectors = np.array([featural_vector])
            flag_v = True
        else:
            vectors = np.append(vectors, [featural_vector], axis=0)

    sout.close()
    featural_data = np.array(vectors)

    # Remove the 'batch' axis.
    # axis0 = featural_data.shape[0]
    # print("axis0: %s " % axis0)
    # axis1 = featural_data.shape[2]
    # print("axis1: %s " % axis1)
    
    # featural_data = featural_data.reshape((axis0, axis1))
    
    return featural_data

ckpt = restore_checkpoint()
model_dict = checkpoint["model_dict"]

