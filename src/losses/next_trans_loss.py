import torch 
import torch.nn as nn

from src.losses.masked_loss import MaskedMSELoss


class NextNumericalFeatureLoss(nn.Module):
    def __init__(self, number):
        super().__init__()
        self.num_criterion = MaskedMSELoss()
        self.number = number
        
    def forward(self, output, batch, mask=None, cat_weights=None, num_weights=None):
        mask = batch['mask'][:, 1:]
        num_pred = output['num_features'][self.number]
        num_trues = batch['num_features'][self.number]
        
        return self.num_criterion(num_pred.squeeze(), num_trues[:, 1:].squeeze(), mask)
    
class NextCatFeatureLoss(nn.Module):
    def __init__(self, number):
        super().__init__()
        self.cat_criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.number = number
        
    def forward(self, output, batch, mask=None, cat_weights=None, num_weights=None):
        mask = batch['mask'][:, 1:]
        cat_pred = output['cat_features'][self.number]
        cat_trues = batch['cat_features'][self.number]
    
        return self.cat_criterion(cat_pred.permute(0, 2, 1), cat_trues[:, 1:])

class NextTimeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.cat_criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.num_criterion = MaskedMSELoss()
        
        
    def forward(self, output, trues, mask=None):
        amnt_out, num_out, need_out = output
        all_amnt_transactions, all_num_transactions, all_code_transactions, next_time_mask = trues
        loss_mask = next_time_mask & mask

        l_amnt = self.num_criterion(amnt_out.squeeze(), all_amnt_transactions.squeeze(), loss_mask)
        l_num = self.num_criterion(num_out.squeeze(), all_num_transactions.squeeze(), loss_mask)
        l_need = (self.cat_criterion(need_out, all_code_transactions) * loss_mask.unsqueeze(-1)).sum()
        
        l_need /= loss_mask.sum()

        return l_amnt + l_need + l_num
        
class NextTransactionLoss(nn.Module):
    def __init__(self, cat_weights=None, num_weights=None, cat_feature_ids=None, num_feature_ids=None):
        super().__init__()
        self.cat_criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.num_criterion = MaskedMSELoss()

        self.cat_weights=cat_weights
        self.num_weights=num_weights
        self.cat_feature_ids=cat_feature_ids
        self.num_feature_ids=num_feature_ids

    def forward(self, output, batch, mask=None, ):
        cat_pred, num_pred = output['cat_features'], output['num_features']
        cat_trues, num_trues = batch['cat_features'], batch['num_features']
        mask = batch['mask'][:, 1:]
        
        res = []
        
        for i, (pred, true) in enumerate(zip(cat_pred, cat_trues)):
            if i not in self.cat_feature_ids:
                continue
            elif self.cat_weights is not None:
                coef = self.cat_weights[i]
            else:
                coef = 1.0
            res.append(coef * self.cat_criterion(pred.permute(0, 2, 1), true[:, 1:]))
        
        for i, (pred, true) in enumerate(zip(num_pred, num_trues)):
            if i not in self.num_feature_ids:
                continue
            
            elif self.num_weights is not None:
                coef = self.num_weights[i]
                
            else:
                coef = 1.0
                
            res.append(coef * self.num_criterion(pred.squeeze(), true[:, 1:].squeeze(), mask))

        return sum(res)
    

def pairwise_distance_torch(embeddings, device):
    """Computes the pairwise distance matrix with numerical stability.
    output[i, j] = || feature[i, :] - feature[j, :] ||_2
    Args:
      embeddings: 2-D Tensor of size [number of data, feature dimension].
    Returns:
      pairwise_distances: 2-D Tensor of size [number of data, number of data].
    """

    # pairwise distance matrix with precise embeddings
    precise_embeddings = embeddings.to(dtype=torch.float32)

    c1 = torch.pow(precise_embeddings, 2).sum(axis=-1)
    c2 = torch.pow(precise_embeddings.transpose(0, 1), 2).sum(axis=0)
    c3 = precise_embeddings @ precise_embeddings.transpose(0, 1)

    c1 = c1.reshape((c1.shape[0], 1))
    c2 = c2.reshape((1, c2.shape[0]))
    c12 = c1 + c2
    pairwise_distances_squared = c12 - 2.0 * c3

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = torch.max(pairwise_distances_squared, torch.tensor([0.]).to(device))
    # Get the mask where the zero distances are at.
    error_mask = pairwise_distances_squared.clone()
    error_mask[error_mask > 0.0] = 1.
    error_mask[error_mask <= 0.0] = 0.

    pairwise_distances = torch.mul(pairwise_distances_squared, error_mask)

    # Explicitly set diagonals to zero.
    mask_offdiagonals = torch.ones((pairwise_distances.shape[0], pairwise_distances.shape[1])) - torch.diag(torch.ones(pairwise_distances.shape[0]))
    pairwise_distances = torch.mul(pairwise_distances.to(device), mask_offdiagonals.to(device))
    return pairwise_distances

def TripletSemiHardLoss(y_true, y_pred, device, margin=1.0):
    """Computes the triplet loss_functions with semi-hard negative mining.
       The loss_functions encourages the positive distances (between a pair of embeddings
       with the same labels) to be smaller than the minimum negative distance
       among which are at least greater than the positive distance plus the
       margin constant (called semi-hard negative) in the mini-batch.
       If no such negative exists, uses the largest negative distance instead.
       See: https://arxiv.org/abs/1503.03832.
       We expect labels `y_true` to be provided as 1-D integer `Tensor` with shape
       [batch_size] of multi-class integer labels. And embeddings `y_pred` must be
       2-D float `Tensor` of l2 normalized embedding vectors.
       Args:
         margin: Float, margin term in the loss_functions definition. Default value is 1.0.
         name: Optional name for the op.
       """

    labels, embeddings = y_true, y_pred

    # Reshape label tensor to [batch_size, 1].
    lshape = labels.shape
    labels = torch.reshape(labels, [lshape[0], 1])

    pdist_matrix = pairwise_distance_torch(embeddings, device)

    # Build pairwise binary adjacency matrix.
    adjacency = torch.eq(labels, labels.transpose(0, 1))
    # Invert so we can select negatives only.
    adjacency_not = adjacency.logical_not()

    batch_size = labels.shape[0]

    # Compute the mask.
    pdist_matrix_tile = pdist_matrix.repeat(batch_size, 1)
    adjacency_not_tile = adjacency_not.repeat(batch_size, 1)

    transpose_reshape = pdist_matrix.transpose(0, 1).reshape(-1, 1)
    greater = pdist_matrix_tile > transpose_reshape

    mask = adjacency_not_tile & greater

    # final mask
    mask_step = mask.to(dtype=torch.float32)
    mask_step = mask_step.sum(axis=1)
    mask_step = mask_step > 0.0
    mask_final = mask_step.reshape(batch_size, batch_size)
    mask_final = mask_final.transpose(0, 1)

    adjacency_not = adjacency_not.to(dtype=torch.float32)
    mask = mask.to(dtype=torch.float32)

    # negatives_outside: smallest D_an where D_an > D_ap.
    axis_maximums = torch.max(pdist_matrix_tile, dim=1, keepdim=True)
    masked_minimums = torch.min(torch.mul(pdist_matrix_tile - axis_maximums[0], mask), dim=1, keepdim=True)[0] + axis_maximums[0]
    negatives_outside = masked_minimums.reshape([batch_size, batch_size])
    negatives_outside = negatives_outside.transpose(0, 1)

    # negatives_inside: largest D_an.
    axis_minimums = torch.min(pdist_matrix, dim=1, keepdim=True)
    masked_maximums = torch.max(torch.mul(pdist_matrix - axis_minimums[0], adjacency_not), dim=1, keepdim=True)[0] + axis_minimums[0]
    negatives_inside = masked_maximums.repeat(1, batch_size)

    semi_hard_negatives = torch.where(mask_final, negatives_outside, negatives_inside)

    loss_mat = margin + pdist_matrix - semi_hard_negatives

    mask_positives = adjacency.to(dtype=torch.float32) - torch.diag(torch.ones(batch_size)).to(device)
    num_positives = mask_positives.sum()

    triplet_loss = (torch.max(torch.mul(loss_mat, mask_positives), torch.tensor([0.]).to(device))).sum() / num_positives
    triplet_loss = triplet_loss.to(dtype=embeddings.dtype)
    return triplet_loss


class TripletLoss(nn.Module):
    def __init__(self, device, margin=0.5):
        super().__init__()
        self.device = device
        self.margin = margin

    def forward(self, input, target, **kwargs):
        return TripletSemiHardLoss(target, input, self.device, self.margin)