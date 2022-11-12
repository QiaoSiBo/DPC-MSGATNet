import torch
from torch.nn.functional import cross_entropy
from torch.nn.modules.loss import _WeightedLoss


EPSILON = 1e-32


class LogNLLLoss(_WeightedLoss):
    __constants__ = ['weight', 'reduction', 'ignore_index']

    def __init__(self, weight=None, size_average=None, reduce=None, reduction=None,
                 ignore_index=-100):
        super(LogNLLLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, y_input, y_target):
        # y_input = torch.log(y_input + EPSILON)
        return cross_entropy(y_input, y_target, weight=self.weight,
                             ignore_index=self.ignore_index)




# def to_one_hot(mask, n_class):  The to_one_hot is used for the mask with shape H * W
#     y_one_hot = torch.zeros((n_class, mask.shape[1], mask.shape[2])).cuda()
#     y_one_hot = y_one_hot.scatter(0, mask.long(), 1).cuda()
#     return y_one_hot

def to_one_hot(predict, mask):   # The to_one_hot is used for the mask with shape B * C * H * W
    y_one_hot = torch.zeros(predict.shape).cuda()
    y_one_hot = y_one_hot.scatter(1, mask.long(), 1).cuda()
    return y_one_hot

def cal_IOU(output, gt):
    '''

    :param output: torch.Tensor of shape (n_batch, n_classes, image.shape)
    :param gt: torch.LongTensor of shape (n_classes, image.shape),,,,one_hot vector
    :return:
    '''

    smooth = 1e-5
    output_ = output > 0.9
    gt_ = gt > 0.9
    intersection = (output_ & gt_).sum()
    union = (output_ | gt_).sum()
    return (float(intersection + smooth)) / (float(union) + smooth)

def classwise_iou(output, gt):
    """
    Args:
        output: torch.Tensor of shape (n_batch, n_classes, image.shape)
        gt: torch.LongTensor of shape (n_batch, image.shape)
    """
    dims = (0, *range(2, len(output.shape)))
    gt = torch.zeros_like(output).scatter_(1, gt[:, None, :], 1)
    intersection = output*gt
    union = output + gt - intersection
    classwise_iou = (intersection.sum(dim=dims).float() + EPSILON) / (union.sum(dim=dims) + EPSILON)

    return classwise_iou


def get_DC(output, gt):
    output = output > 0.9
    gt = gt > 0.9
    inter = torch.sum((output.byte() + gt.byte()) == 2)
    dc = float(2*inter)/(float(torch.sum(output) + torch.sum(gt)) + 1e-6)
    return dc

def get_JS(output, gt):
    output = output > 0.9
    gt = gt > 0.9
    inter = torch.sum((output.byte() + gt.byte()) == 2)
    union = torch.sum((output.byte() + gt.byte()) >= 1)
    js = float(inter) / (float(union) + 1e-6)
    return js

def get_sensitivity(output, gt):
    output = output > 0.9
    gt = gt > 0.9
    TP = ((output==1).byte() + (gt==1).byte()) == 2
    FN = ((output==0).byte() + (gt==1).byte()) == 2
    SE = float(torch.sum(TP)) / (float(torch.sum(TP+FN)) + 1e-6)
    return SE

def get_precision(output, gt):
    output = output > 0.9
    gt = gt > 0.9
    TP = ((output==1).byte() + (gt==1).byte()) == 2
    FP = ((output==1).byte() + (gt==0).byte()) == 2
    PC = float(torch.sum(TP)) / (float(torch.sum(TP+FP)) + 1e-6)

    return PC

def get_F1(output, gt):
    se = get_sensitivity(output, gt)
    pc = get_precision(output, gt)
    f1 = 2*se*pc / (se+pc+1e-6)
    return f1

def classwise_f1(output, gt):
    """
    Args:
        output: torch.Tensor of shape (n_batch, n_classes, image.shape)
        gt: torch.LongTensor of shape (n_batch, image.shape)
        gt: torch.LongTensor of shape (n_classes, image.shape),,,,one_hot vector

    """

    epsilon = 1e-20
    n_classes = output.shape[1]

    output = torch.argmax(output, dim=1)
    gt = torch.argmax(gt, dim=0)
    true_positives = torch.tensor([((output == i) * (gt == i)).sum() for i in range(n_classes)]).float()
    selected = torch.tensor([(output == i).sum() for i in range(n_classes)]).float()
    relevant = torch.tensor([(gt == i).sum() for i in range(n_classes)]).float()

    precision = (true_positives + epsilon) / (selected + epsilon)
    recall = (true_positives + epsilon) / (relevant + epsilon)
    classwise_f1 = 2 * (precision * recall) / (precision + recall)

    classwise_f1 = torch.mean(classwise_f1)

    return classwise_f1


def make_weighted_metric(classwise_metric):
    """
    Args:
        classwise_metric: classwise metric like classwise_IOU or classwise_F1
    """

    def weighted_metric(output, gt, weights=None):

        # dimensions to sum over
        dims = (0, *range(2, len(output.shape)))

        # default weights
        if weights == None:
            weights = torch.ones(output.shape[1]) / output.shape[1]
        else:
            # creating tensor if needed
            if len(weights) != output.shape[1]:
                raise ValueError("The number of weights must match with the number of classes")
            if not isinstance(weights, torch.Tensor):
                weights = torch.tensor(weights)
            # normalizing weights
            weights /= torch.sum(weights)

        classwise_scores = classwise_metric(output, gt).cpu().numpy()

        return classwise_scores 

    return weighted_metric


jaccard_index = make_weighted_metric(classwise_iou)
f1_score = make_weighted_metric(classwise_f1)


if __name__ == '__main__':
    output, gt = torch.zeros(3, 2, 5, 5), torch.zeros(3, 5, 5).long()

