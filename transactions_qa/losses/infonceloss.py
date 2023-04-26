import torch
import torch.nn 
import torch.nn.functional as F


def InfoNCELoss(original_view, corrupted_view):
    # InfoNCE loss

    original_view = F.normalize(original_view, dim=1)
    corrupted_view = F.normalize(corrupted_view, dim=1)

    sim = original_view @ corrupted_view.T
    mask = torch.eye(sim.shape[0]).bool().to(sim.device)

    positive_logits = sim[mask].view(sim.shape[0], 1)
    negative_logits = sim[~mask].view(sim.shape[0], -1)

    logits = torch.cat([positive_logits, negative_logits], dim=1)
    labels = logits.new_zeros(logits.shape[0]).long()
    return F.cross_entropy(logits, labels)