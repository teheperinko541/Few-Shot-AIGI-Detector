""" 
    Prototypical loss from https://arxiv.org/abs/1703.05175. 
"""

from einops import rearrange
import torch.nn.functional as F


def compute_prototypical_loss(inputs, labels, support_num): 
    """ Args:
            inputs: tensors with shape (batch_size, task_samples_num, class_num, features)
            labels: tensors with shape (N, ) or (N, C), where N = batch_size * query_num * class_num
            support_num: capacity of support set used to split inputs data
        
        Returns:
            loss: loss tensor with shape (1, )
            scores: score matrix for calculating cross entropy, which shape (N, C)
    """

    support_set = inputs[:, :support_num, ...]
    query_set = inputs[:, support_num:, ...]

    # compute the barycentres
    prototypes = support_set.mean(dim=1, keepdim=True) # (batch_size, 1, class_num, hidden_dim)

    # compute the distance between each query point and each barycentre, use negative value as scores (so the larger equals to the better)
    scores = - ((rearrange(query_set, 'b q n l -> b (q n) 1 l') - prototypes) ** 2).sum(dim=-1) # (batch_size, query_num * class_num, class_num)
    
    scores = rearrange(scores, 'b n c -> (b n) c')
    loss = F.cross_entropy(scores, labels)

    return loss, scores

