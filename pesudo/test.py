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

def test_func(WSV_model, model, dataloader, criterion, gt, cfg, logger):
    with torch.no_grad():
        WSV_model.eval()
        model.eval()
        reconstruction_errors = []

        for i, (v_input, clip_input, a_input, label) in enumerate(dataloader):
            seq_len = torch.sum(torch.max(torch.abs(v_input), dim=2)[0] > 0, 1)
            v_input = v_input[:, :torch.max(seq_len), :]
            clip_input = clip_input[:, :torch.max(seq_len), :]
            a_input = a_input[:, :torch.max(seq_len), :]
            
            clip_input = pad_tensor(clip_input, torch.max(seq_len))
            
            v_input = v_input.float().cuda(non_blocking=True)
            clip_input = clip_input.float().cuda(non_blocking=True)
            a_input = a_input.float().cuda(non_blocking=True)
            
            if torch.max(seq_len) > 400:
                tmp_pred = torch.zeros(0).cuda()
                for v_in, cl_in, a_in, seq in zip(v_input, clip_input, a_input, seq_len):
                    v_in = v_in.unsqueeze(0)
                    cl_in = cl_in.unsqueeze(0)
                    a_in = a_in.unsqueeze(0)
                    seq = torch.tensor([seq]).cuda()
                    logits, x_v = WSV_model(v_in, cl_in, a_in, seq)
                    real_d = x_v["x"]
                    tmp_pred = torch.cat((tmp_pred, real_d))
                inputs = tmp_pred
            else:
                logits, x_v = WSV_model(v_input, clip_input, a_input, seq_len)
                real_data = x_v["x"]
                inputs = real_data.to('cuda')
            inputs = inputs.permute(0, 2, 1)
            outputs = model(inputs)
            loss = torch.mean((outputs - inputs) ** 2, dim=2).cpu().numpy()
            reconstruction_errors.extend(loss.mean(0))
        
        mean_error = np.mean(reconstruction_errors)
        std_error = np.std(reconstruction_errors)
        threshold = mean_error + std_error
        print(f"Threshold: {threshold}")
        
        pred = reconstruction_errors <= threshold
        
        pred = list(pred)
        fpr, tpr, _ = roc_curve(list(gt), np.repeat(pred, 16))
        roc_auc = auc(fpr, tpr)
        pre, rec, _ = precision_recall_curve(list(gt), np.repeat(pred, 16))
        pr_auc = auc(rec, pre)

        if cfg.dataset == 'ucf-crime':
            return roc_auc, pr_auc
        elif cfg.dataset == 'xd-violence':
            return pr_auc, roc_auc#n_far
        elif cfg.dataset == 'shanghaiTech':
            return roc_auc, pr_auc
        else:
            raise RuntimeError('Invalid dataset.')
        
        
