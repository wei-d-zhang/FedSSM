# FedSSM
{Abstract}
Personalized Federated Learning (PFL) enables collaborative learning across distributed clients while preserving their unique data characteristics. However, personalized models often suffer from catastrophic forgetting, where the local knowledge learned by clients is overwritten by new information from the server model during continuous updates. In this paper, we propose FedSSM, a new framework that leverages state-space models to mitigate forgetting in PFL. By capturing the temporal evolution of local model parameters through hidden states, the framework enhances the retention of critical knowledge across training rounds. Extensive experiments on multiple benchmark datasets demonstrate that FedSSM outperforms various state-of-the-art PFL algorithms, particularly with high data heterogeneity.

# Setup
torchvision >= v0.13  

numpy  
pandas  
np.random.seed(500)  
datasets = {Fashion-MNIST,CIFAR-10,CIFAR-100} \\
model = torchvision.models.resnet18(pretrained=False) \\

# Experiments
FedSSM: python our.py \\
Ditto (ICML 2020)： python Ditto.py \\
FedACG (CVPR 2024)：python FedACG.py \\
FedALA (AAAI 2023): python FedALA.py \\
MOON (CVPR 2021): python MOON.py \\
FedDecorr (ICLR 2023): python FedDecorr.py \\
FedCross (ICDE 2024): python FedCross.py \\
