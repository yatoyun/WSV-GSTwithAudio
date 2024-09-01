import torch
import numpy as np
from sklearn.metrics import auc, roc_curve, confusion_matrix, precision_recall_curve

def cal_false_alarm(gt, preds, threshold=0.5):
    preds = list(preds.cpu().detach().numpy())
    gt = list(gt.cpu().detach().numpy())

    preds = np.repeat(preds, 16)
    preds[preds < threshold] = 0
    preds[preds >= threshold] = 1
    tn, fp, fn, tp = confusion_matrix(gt, preds, labels=[0, 1]).ravel()

    far = fp / (fp + tn)

    return far

def pad_tensor(tensor, max_seqlen):
    batch_size, num_frames, feature_dim = tensor.size()
    padding_length = max_seqlen - num_frames
    if padding_length == 0:
        return tensor
    padded_tensor = torch.cat([tensor, torch.zeros(batch_size, padding_length, feature_dim)], dim=1)
    return padded_tensor

def test_func(dataloader, model, gt, dataset, test_bs):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0)
        ab_pred = torch.zeros(0)

        for i, (v_input, clip_input, a_input, f_input, label) in enumerate(dataloader):
            seq_len = torch.sum(torch.max(torch.abs(v_input), dim=2)[0] > 0, 1)
            v_input = v_input[:, :torch.max(seq_len), :]
            clip_input = clip_input[:, :torch.max(seq_len), :]
            a_input = a_input[:, :torch.max(seq_len), :]
            f_input = f_input[:, :torch.max(seq_len), :]
            
            clip_input = pad_tensor(clip_input, torch.max(seq_len))
            
            v_input = v_input.float().cuda(non_blocking=True)
            clip_input = clip_input.float().cuda(non_blocking=True)
            a_input = a_input.float().cuda(non_blocking=True)
            f_input = f_input.float().cuda(non_blocking=True)
            
            if isinstance(label[0], str):
                label = [1]

            if max(seq_len) < 400:
                logits, _ = model(v_input, clip_input, a_input, f_input, seq_len)
                
                logits = torch.mean(logits, 0)
                logits = logits.squeeze(dim=-1)
                pred = torch.cat((pred, logits.cpu().detach()))
                if sum(label) == len(label):
                    ab_pred = torch.cat((ab_pred, logits.cpu().detach()))
                
            else:
                tmp_pred = torch.zeros(0).cuda()
                for v_in, cl_in, a_in, f_in, seq in zip(v_input, clip_input, a_input, f_input, seq_len):
                    v_in = v_in.unsqueeze(0)
                    cl_in = cl_in.unsqueeze(0)
                    a_in = a_in.unsqueeze(0)
                    f_in = f_in.unsqueeze(0)
                    seq = torch.tensor([seq]).cuda()
                    logits, _ = model(v_in, cl_in, a_in, f_in, seq)
                    tmp_pred = torch.cat((tmp_pred, logits))

                tmp_pred = torch.mean(tmp_pred, 0)
                tmp_pred = tmp_pred.squeeze(dim=-1)
                pred = torch.cat((pred, tmp_pred.cpu().detach()))
                if sum(label) == len(label):
                    ab_pred = torch.cat((ab_pred, tmp_pred.cpu().detach()))

        pred = list(pred)
        fpr, tpr, _ = roc_curve(list(gt), np.repeat(pred, 16))
        roc_auc = auc(fpr, tpr)
        pre, rec, _ = precision_recall_curve(list(gt), np.repeat(pred, 16))
        pr_auc = auc(rec, pre)
        
        ab_pred = list(ab_pred)
        fpr, tpr, _ = roc_curve(list(gt)[:len(ab_pred)*16], np.repeat(ab_pred, 16))
        ab_roc_auc = auc(fpr, tpr)
        
        fpr, tpr, _ = roc_curve(list(gt)[:len(ab_pred)*16], np.repeat(ab_pred, 16))
        ab_roc_auc = auc(fpr, tpr)

        if dataset == 'ucf-crime':
            return roc_auc, ab_roc_auc
        elif dataset == 'xd-violence':
            return pr_auc, roc_auc#n_far
        elif dataset == 'shanghaiTech':
            return roc_auc, ab_roc_auc
        else:
            raise RuntimeError('Invalid dataset.')
