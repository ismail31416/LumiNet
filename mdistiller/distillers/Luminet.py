import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
def perception(logits, epsilon=1e-5):
    """
    perform perception on logits.
    
    Parameters:
    logits (torch.Tensor): A tensor of shape (B, N) where B is the batch size and N is the number of classes.
    epsilon (float): A small constant to avoid division by zero in normalization.
    
    Returns:
    torch.Tensor: perception logits.
    """
    
    
    batch_mean = torch.mean(logits, dim=0, keepdim=True)
    batch_var = torch.var(logits, dim=0, keepdim=True, unbiased=False)
    x_normalized = (logits - batch_mean) / torch.sqrt(batch_var + epsilon)

    
    return x_normalized
    


def luminet_loss(logits_student, logits_teacher, target, alpha, temperature):


    #print('Student')
    stu_batch = perception(logits_student)
    #print('Teacher')
    tea_batch = perception(logits_teacher)
    

    pred_teacher = F.softmax(
        tea_batch/temperature, dim=1
    )
    log_pred_student = F.log_softmax(
        stu_batch/temperature,dim=1
    )
    nckd_loss = F.kl_div(log_pred_student, pred_teacher, reduction='batchmean')
    nckd_loss*=alpha**2

    
    
    return nckd_loss




class Luminet(Distiller):

    def __init__(self, student, teacher, cfg):
        super(Luminet, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.SOLVER.CE_WEIGHT
        self.alpha = cfg.SOLVER.ALPHA
        self.temperature = cfg.SOLVER.T
        self.warmup = cfg.LUMINET.WARMUP

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        loss_ce =  self.ce_loss_weight*F.cross_entropy(logits_student, target)
        loss_dkd = min(kwargs["epoch"] / self.warmup, 1.0) * luminet_loss(
            logits_student,
            logits_teacher,
            target,
            self.alpha,
            self.temperature,
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_dkd,
        }
        return logits_student, losses_dict
