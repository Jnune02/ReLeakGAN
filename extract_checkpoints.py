import pickle as pk
import numpy as np
import torch

from main import restore_checkpoint, generate_samples


print(''' This script should load the checkpoint file, extract the 1st, 2nd, and perhaps 3rd level generators and discriminators, and generates output data to the ./data/ directory. It then loads the respective numpy/tensor objects in that directory and converts them all to text, saving the text files back in the same directory. ''')

def restore_checkpoint(ckpt_path):
    checkpoint = torch.load(ckpt_path)
    return checkpoint

def main():
    use_cuda = torch.cuda.is_available()
    
    checkpoint_l1 = restore_checkpoint("./checkpoints/checkpoint-lvl1.pth.tar")
    model_dict = checkpoint_l1["model_dict"]
    out_fd = "./checkpoints/data/level1.npy"
    
    generate_samples(model_dict, out_fd, batch_size=10,
                            lvl="l1", use_cuda=use_cuda, temperature=1.0)

    data = np.load(out_fd, allow_pickle=True)

    print("Loaded Generated Data as Tensor.")
    print("Shape of loaded Tensor: %s " % list(data.shape))
    print(" Converting to Text...")

    chars_fd = "./data/chars.pkl"
    
    char_fd = open(chars_fd, 'rb')
    char_data = pk.load(char_fd)

    text_data = []
    
    for i in range(data.shape[0]):
        print("Extracting Vector: %s" % i)
        print(data[i].tolist())
        print("Converting to Sentence...")

        tmp1 = data[i].tolist()
        tmp2 = []

        for item in tmp1:
            tmp2.append(char_data[item])

        print(tmp2)
        text_data.append(tmp2)

    out_data = open("./metrics/data/l1_data.txt", "w")
    
    for item in text_data:
        for word in item:
            out_data.write(word + " ")
        out_data.write('\n')
    out_data.close()

    # * * *
    
    # checkpoint_l2 = restore_checkpoint("./checkpoints/checkpoint_l2.pth.tar")
    # model_dict = checkpoint_l2["model_dict"]
    # out_fd = "./checkpoints/data/lvel2.npy"

    # generate_samples(model_dict, out_fd, batch_size=10,
    #                  lvl="l2", use_cuda=use_cuda, temperature=1.0)

    # data = np.load(out_fd, allow_pickle=True)

    # # loaded data should be in the form of a 3 dimensional tensor,
    # # Shape: (Num_Paragraphs x Sentences per Paragraph x Words per Sentence)

    # print("Loaded Generated Data as Tensor.")
    # print("Shape of loaded Tensor: %s " % list(data.shape))
    # print(" Converting to Text...")

    # par_data = []
    
    # for i in range(data.shape[0]):
    #     print("Extracting Vector...")
    #     print("Converting to Text...")

    #     text_data = []
        
    #     for j in range(data.shape[1]):
    #         tmp1 = data[i][j].tolist()
    #         tmp2 = []

    #         for item in tmp1:
    #             tmp2.append(char_data[item])

    #         tmp2.append("STOP.")
    #         text_data.append(tmp2)

    #     par_data.append(text_data)

    # out_data = open("./metrics/data/l1_data.txt", "w")
    # out_data.write(par_data)
    # out_data.close
    
    
if __name__ == "__main__":
    main()
