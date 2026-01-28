import torch
from matplotlib import pyplot as plt

def load_checkpoint(path):
        """Carica checkpoint"""
        ckpt = torch.load(path)
        history = ckpt['history']

        #print("validation loss policy: ", history["val_policy"])
        print("training loss policy: ", history["train_policy"])
        #plt.plot(history["train_policy"])
        #plt.show()

resume_from = "./GCN/epoch_050.pt"  # o epoch_010.pt
loaded_epoch = load_checkpoint(resume_from)

resume_from = "./MPNN/epoch_110.pt"  # o epoch_010.pt
loaded_epoch = load_checkpoint(resume_from)