import torch

def split_process(batch, splitter):
    res = {}


    seq_len = batch['mask'].shape[1]
    local_date = torch.arange(seq_len)
    if splitter is not None:
        indexes = splitter.split(local_date)
        pad_size = max([len(ixs) for ixs in indexes])
    
    for k, v in batch.items():
        if k in ['num_features', 'cat_features'] and splitter is not None:
            new_v = []
            for elem in v:
                tmp = []
                for i, ixs in enumerate(indexes):
                    to_tmp = elem[:, ixs]
                    if to_tmp.shape[1] < pad_size:
                        to_tmp = torch.cat([
                            to_tmp, torch.zeros(to_tmp.shape[0], pad_size - to_tmp.shape[1], dtype=torch.int)
                        ], axis=1)
                    tmp.append(to_tmp)
                new_v.append(torch.cat(tmp, dim=0))
            new_v = torch.stack(new_v, dim=0)
        elif k == 'meta_features' and splitter is not None:
            new_v = v.repeat(1, len(indexes))
        else:
            new_v = v 
        res[k] = new_v
    res['mask'] = res['cat_features'][0] != 0
    return res