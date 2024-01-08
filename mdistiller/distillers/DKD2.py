import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
def batch_norm(logits, epsilon=1e-5):
    """
    Perform batch normalization on logits.
    
    Parameters:
    logits (torch.Tensor): A tensor of shape (B, N) where B is the batch size and N is the number of classes.
    epsilon (float): A small constant to avoid division by zero in normalization.
    
    Returns:
    torch.Tensor: The batch-normalized logits.
    """
    
    # Calculate the mean and variance along the batch dimension
    #mean = torch.mean(logits, dim=0, keepdim=True)
    #var = torch.var(logits, dim=0, keepdim=True)
    
    # Perform batch normalization
    #normalized_logits = (logits - mean) / torch.sqrt(var + epsilon)
    # Compute the mean and variance for this mini-batch
    batch_mean = torch.mean(logits, dim=0, keepdim=True)
    batch_var = torch.var(logits, dim=0, keepdim=True, unbiased=False)
    
    # Normalize the batch
    x_normalized = (logits - batch_mean) / torch.sqrt(batch_var + epsilon)
    
    return x_normalized
    
    

def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)

    stu_batch = batch_norm(logits_student)
    tea_batch = batch_norm(logits_teacher)

    pred_teacher_part2 = F.softmax(
        torch.sigmoid(tea_batch)/2, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        torch.sigmoid(stu_batch)/2, dim=1
    )
    nckd_loss = F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='batchmean')
    nckd_loss*=18**2



    logits_student = torch.sigmoid(stu_batch)*logits_student
    logits_teacher = torch.sigmoid(tea_batch)*logits_teacher
    pred_student = F.softmax(logits_student/10, dim=1)
    pred_teacher = F.softmax(logits_teacher/10, dim=1)
    
    log_pred_student = torch.log(pred_student)
    tckd_loss = F.kl_div(log_pred_student,pred_teacher,reduction='batchmean')
    tckd_loss*=25**2

    
    
    return 1.0*tckd_loss+3*0.2*nckd_loss


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


class DKD(Distiller):
    """Decoupled Knowledge Distillation(CVPR 2022)"""

    def __init__(self, student, teacher, cfg):
        super(DKD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.DKD.CE_WEIGHT
        self.alpha = cfg.DKD.ALPHA
        self.beta = cfg.DKD.BETA
        self.temperature = cfg.DKD.T
        self.warmup = cfg.DKD.WARMUP

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        loss_ce =  F.cross_entropy(logits_student, target)
        loss_dkd = min(kwargs["epoch"] / self.warmup, 1.0) * dkd_loss(
            logits_student,
            logits_teacher,
            target,
            self.alpha,
            self.beta,
            self.temperature,
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_dkd,
        }
        return logits_student, losses_dict
