"""
Merging algorithm for Fourier Token Merging

Most of the code for Fourier Token Merging is dervied from this repo: 
https://github.com/dbolya/tomesd


"""

import torch
import pywt
from torch.cuda.amp import autocast

from pytorch_wavelets import DWT1DForward
from typing import Tuple, Callable
from timeit import default_timer as timer


def do_nothing(x: torch.Tensor, mode:str=None):
    return x


def mps_gather_workaround(input, dim, index):
    if input.shape[-1] == 1:
        return torch.gather(
            input.unsqueeze(-1),
            dim - 1 if dim < 0 else dim,
            index.unsqueeze(-1)
        ).squeeze(-1)
    else:
        return torch.gather(input, dim, index)


def bipartite_soft_matching_random2d(metric: torch.Tensor,
                                     w: int, h: int, sx: int, sy: int, r: int,
                                     ratio: float = 0.5, # add existed
                                     use_dft: bool = False,  # new
                                     trunc_ratio: float = None, # new
                                     high_weight: float = 1.0, # new
                                     low_weight: float = 1.0,  # new
                                     no_rand: bool = False,
                                     generator: torch.Generator = None) -> Tuple[Callable, Callable]:
    """
    Partitions the tokens into src and dst and merges r tokens from src to dst.
    Dst tokens are partitioned by choosing one randomy in each (sx, sy) region.

    Args:
     - metric [B, N, C]: metric to use for similarity
     - w: image width in tokens
     - h: image height in tokens
     - sx: stride in the x dimension for dst, must divide w
     - sy: stride in the y dimension for dst, must divide h
     - r: number of tokens to remove (by merging)
     - no_rand: if true, disable randomness (use top left corner only)
     - rand_seed: if no_rand is false, and if not None, sets random seed.
    """
    B, N, _ = metric.shape

    #start_matching = timer()

    if r <= 0:
        return do_nothing, do_nothing

    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather
    
    with torch.no_grad():
        hsy, wsx = h // sy, w // sx

        # For each sy by sx kernel, randomly assign one token to be dst and the rest src
        if no_rand:
            rand_idx = torch.zeros(hsy, wsx, 1, device=metric.device, dtype=torch.int64)
        else:
            rand_idx = torch.randint(sy*sx, size=(hsy, wsx, 1), device=generator.device, generator=generator).to(metric.device)
        
        # The image might not divide sx and sy, so we need to work on a view of the top left if the idx buffer instead
        idx_buffer_view = torch.zeros(hsy, wsx, sy*sx, device=metric.device, dtype=torch.int64)
        idx_buffer_view.scatter_(dim=2, index=rand_idx, src=-torch.ones_like(rand_idx, dtype=rand_idx.dtype))
        idx_buffer_view = idx_buffer_view.view(hsy, wsx, sy, sx).transpose(1, 2).reshape(hsy * sy, wsx * sx)

        # Image is not divisible by sx or sy so we need to move it into a new buffer
        if (hsy * sy) < h or (wsx * sx) < w:
            idx_buffer = torch.zeros(h, w, device=metric.device, dtype=torch.int64)
            idx_buffer[:(hsy * sy), :(wsx * sx)] = idx_buffer_view
        else:
            idx_buffer = idx_buffer_view

        # We set dst tokens to be -1 and src to be 0, so an argsort gives us dst|src indices
        rand_idx = idx_buffer.reshape(1, -1, 1).argsort(dim=1)

        # We're finished with these
        del idx_buffer, idx_buffer_view

        # rand_idx is currently dst|src, so split them
        num_dst = hsy * wsx
        a_idx = rand_idx[:, num_dst:, :] # src
        b_idx = rand_idx[:, :num_dst, :] # dst

        def split(x):
            C = x.shape[-1]
            src = gather(x, dim=1, index=a_idx.expand(B, N - num_dst, C))
            dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
            return src, dst

        #use_dft = False
        #use_dft = True
        #print('use_dft, trunc_ratio : ', use_dft, trunc_ratio)

        


        if use_dft:
            
            metric_fp32 = metric.float()
            dft_metric = torch.fft.fft(metric_fp32, dim=-1).real

            trunc_len = int(dft_metric.shape[-1] * trunc_ratio)  # 例如保留一半，也可以自定义
            

            # original truncation
            dft_metric_truncated = dft_metric[..., :trunc_len]

            dff_metric = dft_metric_truncated / dft_metric_truncated.norm(dim=-1, keepdim=True)

            dff_metric = dff_metric.to(metric.dtype)

            a, b = split(dff_metric)


        else:
            # Cosine similarity between A and B
            metric = metric / metric.norm(dim=-1, keepdim=True)
            a, b = split(metric)
        
        scores = a @ b.transpose(-1, -2)
        

       
        r = min(a.shape[1], r)

        # Find the most similar greedily
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)

        # output the number active clusters (clusters in the dst set that have src tokens merging into it) #
        
        B, R, _ = dst_idx.shape

        metric_norm = metric / metric.norm(dim=-1, keepdim=True)  # [B, N, C]

        log_analysis = True
        log_analysis = False

        if log_analysis:

            import csv
            

            with open(f"theoretical_analysis/class_4/cluster_metrics_dft{use_dft}_tome_ratio{ratio}_trunc_ratio_{trunc_ratio}.csv", "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                
                for b in range(B):
                    unique_clusters = dst_idx[b].flatten().unique()
                    batch_ssd = []
                    batch_custom = []

                    for cluster_id in unique_clusters:
                        sel_src = (dst_idx[b].flatten() == cluster_id).nonzero(as_tuple=False).squeeze(-1)
                        indices = torch.cat([
                            torch.tensor([cluster_id], device=metric.device),
                            sel_src + src_idx.new_zeros(1)
                        ]).long()

                        

                        #### custom metrics ##########
                        X = metric_norm[b, indices].float()    
                        mu = X.mean(dim=0, keepdim=True)       

                        sum_sq = (X ** 2).sum()               
                        sum_sq_sq = sum_sq ** 2               

                        # compute SSD
                        ssd = ((X - mu) ** 2).sum()            
                        batch_ssd.append(ssd)

                        #
                        custom_metric = sum_sq_sq * ssd
                        #ssd = sum_sq_sq * ssd
                        batch_custom.append(custom_metric)

                    ssd_tensor = torch.tensor(batch_ssd, device=metric.device, dtype=torch.float32)
                    batch_tensor = torch.tensor(batch_custom, device=metric.device, dtype=torch.float32)

                    avg_ssd    = ssd_tensor.mean().item()
                    min_ssd    = ssd_tensor.min().item()
                    max_ssd    = ssd_tensor.max().item()
                    median_ssd = ssd_tensor.median().item()
                    std_ssd    = ssd_tensor.std(unbiased=True).item()
                    q25_ssd    = torch.quantile(ssd_tensor, 0.25).item()
                    q75_ssd    = torch.quantile(ssd_tensor, 0.75).item()

                    ###  similarity within clusters ###
                    src_idx_squeezed = src_idx.squeeze(-1)  # shape: [B, r]
                    dst_idx_squeezed = dst_idx.squeeze(-1)  # shape: [B, r]

                    cluster_sims = {}
                    for i in range(src_idx_squeezed.shape[1]):
                        s_idx = int(src_idx_squeezed[b, i].item())
                        d_idx = int(dst_idx_squeezed[b, i].item())
                        sim_val = scores[b, s_idx, d_idx].item()
                        cluster_sims.setdefault(d_idx, []).append(sim_val)


                    all_sims = [v for sims in cluster_sims.values() for v in sims]
                    if all_sims:
                        mean_sim = sum(all_sims) / len(all_sims)
                    else:
                        mean_sim = 0.0

                    ### num of clusters ####
                    unique_clusters = dst_idx[b].flatten().unique()

                    writer.writerow([b,
                                    mean_sim,
                                    unique_clusters.numel(),
                                    ssd_tensor.mean().item(),
                                    batch_tensor.mean().item(),
                                    ])

                    
        

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = split(x)
        n, t1, c = src.shape
        
        unm = gather(src, dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = gather(src, dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        _, _, c = unm.shape

        src = gather(dst, dim=-2, index=dst_idx.expand(B, r, c))

        # Combine back to the original shape
        out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.expand(B, num_dst, c), src=dst)
        out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=unm_idx).expand(B, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=src_idx).expand(B, r, c), src=src)

        return out

    #end_matching = timer()
    #print("Matching time: ", end_matching - start_matching)

    return merge, unmerge
