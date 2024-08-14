import torch
import numpy as np
# from utils.general import box_iou
from ultralytics.utils.metrics import bbox_iou
# reference: https://github.com/dongdonghy/repulsion_loss_pytorch/blob/master/repulsion_loss.py
def pairwise_bbox_iou(box1, box2, box_format = 'xywh'):
    if box_format == 'xyxy':
        lt = torch.max(box1[:, None, :2], box2[:, :2])
        rb = torch.min(box1[:, None, 2:], box2[:, 2:])
        area_1 = torch.prod(box1[:, 2:] - box1[:, :2], 1)
        area_2 = torch.prod(box1[:, 2:] - box1[:, :2], 1)
    elif box_format == 'xywh':
        lt = torch.max(
            (box1[:, None, :2] - box1[:, None, :2] / 2),
            (box2[:, :2] + box2[:, 2:] / 2),
        )
        rb = torch.min(
            (box1[:, None, :2], + box1[:, None, 2:] / 2),
            (box2[:, :2], + box2[:, 2:] / 2),
        )
        area_1 = torch.prod(box1[:, 2:], 1)
        area_2 = torch.prod(box2[:, 2:], 1)
    valid = (lt < rb).type(lt.type()).prod(dim=2)
    inter = torch.prod(rb - lt, 2) * valid
    return inter / (area_1[:, None] + area_2 - inter)


def IoG(gt_box, pre_box):
    inter_xmin = torch.max(gt_box[:, 0], pre_box[:, 0])
    inter_ymin = torch.max(gt_box[:, 1], pre_box[:, 1])
    inter_xmax = torch.min(gt_box[:, 2], pre_box[:, 2])
    inter_ymax = torch.min(gt_box[:, 3], pre_box[:, 3])
    Iw = torch.clamp(inter_xmax - inter_xmin, min=0)
    Ih = torch.clamp(inter_ymax - inter_ymin, min=0)
    I = Iw * Ih
    G = ((gt_box[:, 2] - gt_box[:, 0]) * (gt_box[:, 3] - gt_box[:, 1])).clamp(1e-6)
    return I / G

def smooth_ln(x, deta=0.5):
    return torch.where(
        torch.le(x, deta),
        -torch.log(1 - x),
        ((x - deta) / (1 - deta)) - np.log(1 - deta)
    )

# YU 添加了detach，减小了梯度对gpu的占用
# def repulsion_loss_torch(pbox, gtbox, deta=0.5, pnms=0.1, gtnms=0.1, x1x2y1y2=False):
#     repgt_loss = 0.0
#     repbox_loss = 0.0
#     pbox = pbox.detach()
#     gtbox = gtbox.detach()
#     gtbox_cpu = gtbox.cuda().data.cpu().numpy()
#     pgiou = bbox_iou(pbox, gtbox, xywh=x1x2y1y2)
#     pgiou = pgiou.cuda().data.cpu().numpy()
#     ppiou = bbox_iou(pbox, pbox, xywh=x1x2y1y2)
#     ppiou = ppiou.cuda().data.cpu().numpy()
#     # t1 = time.time()
#     len = pgiou.shape[0]
#     for j in range(len):
#         for z in range(j, len):
#             ppiou[j, z] = 0
#             # if int(torch.sum(gtbox[j] == gtbox[z])) == 4:
#             # if int(torch.sum(gtbox_cpu[j] == gtbox_cpu[z])) == 4:
#             # if int(np.sum(gtbox_numpy[j] == gtbox_numpy[z])) == 4:
#             if (gtbox_cpu[j][0]==gtbox_cpu[z][0]) and (gtbox_cpu[j][1]==gtbox_cpu[z][1]) and (gtbox_cpu[j][2]==gtbox_cpu[z][2]) and (gtbox_cpu[j][3]==gtbox_cpu[z][3]):
#                 pgiou[j, z] = 0
#                 pgiou[z, j] = 0
#                 ppiou[z, j] = 0

#     # t2 = time.time()
#     # print("for cycle cost time is: ", t2 - t1, "s")
#     pgiou = torch.from_numpy(pgiou).cuda().detach()
#     ppiou = torch.from_numpy(ppiou).cuda().detach()
#     # repgt
#     max_iou, argmax_iou = torch.max(pgiou, 1)
#     pg_mask = torch.gt(max_iou, gtnms)
#     num_repgt = pg_mask.sum()
#     if num_repgt > 0:
#         iou_pos = pgiou[pg_mask, :]
#         max_iou_sec, argmax_iou_sec = torch.max(iou_pos, 1)
#         pbox_sec = pbox[pg_mask, :]
#         gtbox_sec = gtbox[argmax_iou_sec, :]
#         IOG = IoG(gtbox_sec, pbox_sec)
#         repgt_loss = smooth_ln(IOG, deta)
#         repgt_loss = repgt_loss.mean()

#     # repbox
#     pp_mask = torch.gt(ppiou, pnms)  # 防止nms为0, 因为如果为0,那么上面的for循环就没有意义了 [N x N] error
#     num_pbox = pp_mask.sum()
#     if num_pbox > 0:
#         repbox_loss = smooth_ln(ppiou, deta)
#         repbox_loss = repbox_loss.mean()
#     # mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
#     # print(mem)
#     torch.cuda.empty_cache()

#     return repgt_loss, repbox_loss
# def repulsion_loss(pbox, gtbox, fg_mask, sigma_repgt = 0.9, sigma_repbox = 0, pnms = 0, gtnms = 0):
#     loss_repgt = torch.zeros(1).to(pbox.device)
#     loss_repgt = torch.zeros(1).to(pbox.device)
#     bbox_mask = fg_mask.unsqueeze(-1).repeat([1,1,4])
#     bs = 0
#     pbox = pbox.detach()
#     gtbox = gtbox.detach()
#     for idx in range(pbox.shape[0]):
#         num_pos = bbox_mask[idx].sum()
#         if num_pos <= 0:
#             continue
#         _pbox_pos = torch.masked_select(pbox[idx], bbox_mask[idx].reshape([-1, 4]))
#         _gtbox_pos = torch.masked_select(gtbox[idx], bbox_mask[idx].reshape([-1, 4]))
#         bs += 1
#         pgiou = pairwise_bbox_iou(_pbox_pos, _gtbox_pos, box_format= 'xyxy')
#         ppiou = pairwise_bbox_iou(_pbox_pos, _pbox_pos, box_format='xyxy')
#         pgiou = pgiou.cuda().data.cpu().numpy()
#         ppiou = ppiou.cuda().data.cpu().numpy()
#         _gtbox_pos_cpu = _gtbox_pos.cuda().data.cpu().numpy()
        
#         for j in range(pgiou.shape[0]):
#             for z in range(j, pgiou.shape[0]):
#                 ppiou[j, z] = 0
#                 if(_gtbox_pos_cpu[j][0] == _gtbox_pos_cpu[z][0]) and (_gtbox_pos_cpu[j][1] == _gtbox_pos_cpu[z][1]) \
#                     and (_gtbox_pos_cpue[j][2] == _gtbox_pose_cpu[z][2]) 
                    
def repulsion_loss(pbox, gtbox, fg_mask, sigma_regpt = 0.9, sigma_repbox = 0, pnms = 0, gtnms = 0):  # nms=0
    loss_regpt = torch.zeros(1).to(pbox.device)
    loss_repbox = torch.zeros(1).to(pbox.device)
    bbox_mask = fg_mask.unsqueeze(-1).repeat([1, 1, 4])
    bs = 0
    pbox = pbox.detach()
    gtbox = gtbox.detach()
    for idx in range(pbox.shape[0]):
        num_pos = bbox_mask[idx].sum()
        if num_pos <= 0:
            continue
        _pbox_pos = torch.masked_select(pbox[idx], bbox_mask[idx]).reshape([-1, 4])
        _gtbox_pos = torch.masked_select(gtbox[idx], bbox_mask[idx]).reshape([-1, 4])
        bs += 1
        pgiou = pairwise_bbox_iou(_pbox_pos, _gtbox_pos, box_format='xyxy')
        ppiou = pairwise_bbox_iou(_pbox_pos, _pbox_pos, box_format='xyxy')
        pgiou = pgiou.cuda().data.cpu().numpy()
        ppiou = ppiou.cuda().data.cpu().numpy()
        _gtbox_pos_cpu = _gtbox_pos.cuda().data.cpu().numpy()
        
        for j in range(pgiou.shape[0]):
            for z in range(j, pgiou.shape[0]):
                ppiou[j, z] = 0
                if (_gtbox_pos_cpu[j][0] == _gtbox_pos_cpu[z][0]) and (_gtbox_pos_cpu[j][1] == _gtbox_pos_cpu[z][1]) \
                        and (_gtbox_pos_cpu[j][2] == _gtbox_pos_cpu[z][2]) and (_gtbox_pos_cpu[j][3] == _gtbox_pos_cpu[z][3]):
                    pgiou[j, z] = 0
                    pgiou[z, j] = 0
                    ppiou[z, j] = 0
        pgiou = torch.from_numpy(pgiou).to(pbox.device).cuda().detach()
        ppiou = torch.from_numpy(ppiou).to(pbox.device).cuda().detach()
        max_iou, _ = torch.max(pgiou, 1)
        pg_mask = torch.gt(max_iou, gtnms)
        num_regpt = pg_mask.sum()
        if num_regpt > 0:
            pgiou_pos = pgiou[pg_mask, :]
            _, argmax_iou_sec = torch.max(pgiou_pos, 1)
            pbox_sec = _pbox_pos[pg_mask, :]
            gtbox_sec = _gtbox_pos[argmax_iou_sec, :]
            IOG = IoG(gtbox_sec, pbox_sec)
            loss_regpt += smooth_ln(IOG, sigma_regpt).mean()
        pp_mask = torch.gt(ppiou, pnms)
        num_pbox = pp_mask.sum()
        if num_pbox > 0:
            loss_repbox += smooth_ln(ppiou, sigma_repbox).mean()
    loss_regpt /= bs
    loss_repbox /= bs
    torch.cuda.empty_cache()
    return loss_regpt.squeeze(0), loss_repbox.squeeze(0)
