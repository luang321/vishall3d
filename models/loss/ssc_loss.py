import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def KL_sep(p, target):
    """
    KL divergence on nonzeros classes
    """
    nonzeros = target != 0
    nonzero_p = p[nonzeros]
    kl_term = F.kl_div(torch.log(nonzero_p), target[nonzeros], reduction="sum")
    return kl_term

'''
def geo_scal_loss(pred, ssc_target,masks = [255,]):

    # Get softmax probabilities
    pred = F.softmax(pred, dim=1)

    # Compute empty and nonempty probabilities
    empty_probs = pred[:,0] #pred[:, 0, :, :, :]
    nonempty_probs = 1 - empty_probs

    # Remove unknown voxels
    mask = (ssc_target == 255)#ssc_target != 255
    for m in masks:
        mask = mask|(ssc_target == m)
    mask = ~mask
    nonempty_target = ssc_target != 0
    nonempty_target = nonempty_target[mask].float()
    nonempty_probs = nonempty_probs[mask]
    empty_probs = empty_probs[mask]

    intersection = (nonempty_target * nonempty_probs).sum()
    precision = intersection / nonempty_probs.sum()
    recall = intersection / nonempty_target.sum()
    spec = ((1 - nonempty_target) * (empty_probs)).sum() / (1 - nonempty_target).sum()
    return (
        F.binary_cross_entropy(precision, torch.ones_like(precision))
        + F.binary_cross_entropy(recall, torch.ones_like(recall))
        + F.binary_cross_entropy(spec, torch.ones_like(spec))
    )

def binary_geo_scal_loss(pred, ssc_target):

    # Get softmax probabilities
    # Compute empty and nonempty probabilities
    empty_probs = 1- pred 
    nonempty_probs = pred
    # Remove unknown voxels
    mask = (ssc_target == 255)#ssc_target != 255
    mask = ~mask

    nonempty_target = ssc_target != 0
    nonempty_target = nonempty_target[mask].float()
    nonempty_probs = nonempty_probs[mask]
    empty_probs = empty_probs[mask]

    intersection = (nonempty_target * nonempty_probs).sum()
    precision = intersection / nonempty_probs.sum()
    recall = intersection / nonempty_target.sum()
    spec = ((1 - nonempty_target) * (empty_probs)).sum() / (1 - nonempty_target).sum()
    return (
        F.binary_cross_entropy(precision, torch.ones_like(precision))
        + F.binary_cross_entropy(recall, torch.ones_like(recall))
        + F.binary_cross_entropy(spec, torch.ones_like(spec))
    )


def sem_scal_loss(pred, ssc_target,masks = [255,]):
    # Get softmax probabilities
    pred = F.softmax(pred, dim=1)
    loss = 0
    count = 0
    mask = (ssc_target == 255)#ssc_target != 255
    for m in masks:
        #print("AsF",mask.shape,mask.sum())
        mask = mask|(ssc_target == m)
        #print("aaaaF",mask.shape,mask.sum())
    mask = ~mask
    n_classes = pred.shape[1]
    for i in range(0, n_classes):

        # Get probability of class i
        p = pred[:,i]#[:, i, :, :, :]
        
        # Remove unknown voxels
        target_ori = ssc_target
        p = p[mask]
        target = ssc_target[mask]

        completion_target = torch.ones_like(target)
        completion_target[target != i] = 0
        completion_target_ori = torch.ones_like(target_ori).float()
        completion_target_ori[target_ori != i] = 0
        if torch.sum(completion_target) > 0:
            count += 1.0
            nominator = torch.sum(p * completion_target)
            loss_class = 0
            if torch.sum(p) > 0:
                precision = nominator / (torch.sum(p))
                loss_precision = F.binary_cross_entropy(
                    precision, torch.ones_like(precision)
                )
                loss_class += loss_precision
            if torch.sum(completion_target) > 0:
                recall = nominator / (torch.sum(completion_target))
                loss_recall = F.binary_cross_entropy(recall, torch.ones_like(recall))
                loss_class += loss_recall
            if torch.sum(1 - completion_target) > 0:
                specificity = torch.sum((1 - p) * (1 - completion_target)) / (
                    torch.sum(1 - completion_target)
                )
                loss_specificity = F.binary_cross_entropy(
                    specificity, torch.ones_like(specificity)
                )
                loss_class += loss_specificity
            loss += loss_class
    return loss / count


def CE_ssc_loss(pred, target, class_weights,masks = [255,]):
    """
    :param: prediction: the predicted tensor, must be [BS, C, H, W, D]
    """
    #print("pre" ,(target.flatten()==255).sum())
    target_ = target.clone()
    for m in masks:
        target_[target_==m] = 255
    #print("aft" ,(target.flatten()==255).sum(),(target_.flatten()==255).sum())
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, ignore_index=255, reduction="mean"
    )
    loss = criterion(pred, target_.long())

    return loss
'''


def geo_scal_loss(pred, ssc_target,masks = [255,],b = 0):

    # Get softmax probabilities
    pred = F.softmax(pred, dim=1)

    # Compute empty and nonempty probabilities
    empty_probs = pred[:,0] #pred[:, 0, :, :, :]
    nonempty_probs = 1 - empty_probs

    # Remove unknown voxels
    mask = (ssc_target == 255)#ssc_target != 255
    for m in masks:
        mask = mask|(ssc_target == m)
    mask = ~mask
    nonempty_target = ssc_target != 0
    nonempty_target = nonempty_target[mask].float()
    nonempty_probs = nonempty_probs[mask]
    empty_probs = empty_probs[mask]

    intersection = (nonempty_target * nonempty_probs).sum()
    precision = intersection / nonempty_probs.sum()
    recall = intersection / nonempty_target.sum()
    spec = ((1 - nonempty_target) * (empty_probs)).sum() / (1 - nonempty_target).sum()

    loss_p = F.binary_cross_entropy(precision, torch.ones_like(precision),reduction = "none")
    loss_r = F.binary_cross_entropy(recall, torch.ones_like(recall),reduction = "none")
    loss_s = F.binary_cross_entropy(spec, torch.ones_like(spec),reduction = "none")
    loss_p = (loss_p - b).abs() + b
    loss_r = (loss_r - b).abs() + b
    loss_s = (loss_s - b).abs() + b
    return (loss_p + loss_r + loss_s).mean()


def sem_scal_loss(pred, ssc_target,masks = [255,],b = 0 ):
    # Get softmax probabilities
    pred = F.softmax(pred, dim=1)
    loss = 0
    count = 0
    mask = (ssc_target == 255)#ssc_target != 255
    for m in masks:
        #print("AsF",mask.shape,mask.sum())
        mask = mask|(ssc_target == m)
        #print("aaaaF",mask.shape,mask.sum())
    mask = ~mask
    n_classes = pred.shape[1]

    for i in range(0, n_classes):

        # Get probability of class i
        p = pred[:,i]#[:, i, :, :, :]
        
        # Remove unknown voxels
        target_ori = ssc_target
        p = p[mask]
        target = ssc_target[mask]

        completion_target = torch.ones_like(target)
        completion_target[target != i] = 0
        completion_target_ori = torch.ones_like(target_ori).float()
        completion_target_ori[target_ori != i] = 0
        if torch.sum(completion_target) > 0:
            count += 1.0
            nominator = torch.sum(p * completion_target)
            loss_class = 0
            if torch.sum(p) > 0:
                precision = nominator / (torch.sum(p))
                loss_precision = F.binary_cross_entropy(
                    precision, torch.ones_like(precision),reduction = "none"
                )
                loss_precision = (loss_precision - b).abs() + b
                loss_class += loss_precision.mean()
            if torch.sum(completion_target) > 0:
                recall = nominator / (torch.sum(completion_target))
                loss_recall = F.binary_cross_entropy(recall, torch.ones_like(recall),reduction = "none")
                loss_recall = (loss_recall - b).abs() + b
                loss_class += loss_recall.mean()
            if torch.sum(1 - completion_target) > 0:
                specificity = torch.sum((1 - p) * (1 - completion_target)) / (
                    torch.sum(1 - completion_target)
                )
                loss_specificity = F.binary_cross_entropy(
                    specificity, torch.ones_like(specificity),reduction = "none"
                )
                loss_specificity = (loss_specificity - b).abs() + b
                loss_class += loss_specificity.mean()
            loss += loss_class
    return loss / count


def CE_ssc_loss(pred, target, class_weights,masks = [255,],b = 0):
    """
    :param: prediction: the predicted tensor, must be [BS, C, H, W, D]
    """
    #print("pre" ,(target.flatten()==255).sum())
    target_ = target.clone()
    for m in masks:
        target_[target_==m] = 255
    #print("aft" ,(target.flatten()==255).sum(),(target_.flatten()==255).sum())
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, ignore_index=255, reduction="none"
    )
    loss = criterion(pred, target_.long())
    #print((loss<0).sum())
    loss = (loss - b).abs() + b
    loss = loss[target_!= 255]
    weight = class_weights[target_[target_!= 255]]
    loss = loss.sum()/(weight.sum())
    '''
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, ignore_index=255,reduction = "none"
    )
    l2 = criterion(pred, target_.long())
    print(l2.shape)
    l2 = l2[target_!= 255]
    weight = class_weights[target_[target_!= 255]]
    print(l2.shape,l2.sum()/(weight.sum()))'''
    return loss


