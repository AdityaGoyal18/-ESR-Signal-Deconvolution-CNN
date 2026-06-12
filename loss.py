import torch

def esr_loss(pred,target,input_signal):

    loss_r1=torch.mean((pred[:,0,:]-target[:,0,:])**2)
    loss_r2=torch.mean((pred[:,1,:]-target[:,1,:])**2)

    radical_loss=loss_r1+loss_r2

    recon=torch.sum(pred,dim=1)

    physics=torch.mean((recon-input_signal)**2)

    return radical_loss+0.5*physics