import torch

def slice_time_series(x, n_windows):
    """
    Slice a batched time series (batch, time, channel) into N equal-size windows
    with minimal overlap along the time dimension.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (batch, time, channel).
    n_windows : int
        Number of windows to generate.

    Returns
    -------
    windows : torch.Tensor
        Output tensor of shape (batch, window_size * n_windows, channel).
    """
    if x.ndim != 3:
        raise ValueError("Input must have shape (batch, time, channel)")

    B, T, C = x.shape

    if n_windows <= 0 or n_windows > T:
        raise ValueError("n_windows must be between 1 and the time dimension size")

    # Window size = ceil(T / N) to cover the full time series
    win_size = (T + n_windows - 1) // n_windows  # Equivalent to ceil(T / n_windows)

    # Stride so windows are evenly spread
    stride = (T - win_size) / max(n_windows - 1, 1)


    windows = []
    for i in range(n_windows):
        start = int(round(i * stride))
        end = start + win_size
        windows.append(x[:, start:end, :].unsqueeze(1))  # shape: (batch, window_size, channel)

    # Concatenate along the time dimension
    return torch.cat(windows, dim=1).reshape(B*n_windows, -1, C)


def slice_and_split_with_sub(x, x_sub, n_windows):
    """
    Slice a batched time series (batch, time, channel) into N equal-size windows
    with minimal overlap along the time dimension, while maintaining alignment
    with the subject information tensor.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (batch, time, channel).
    x_sub : torch.Tensor
        Subject ID tensor of shape (batch, 1, 1, subject_dim).
    n_windows : int
        Number of windows to generate.

    Returns
    -------
    x_sliced : torch.Tensor
        Sliced tensor of shape (batch * n_windows, window_size, channel).
    x_sub_sliced : torch.Tensor
        Sliced subject tensor of shape (batch * n_windows, 1, 1, subject_dim).
    """
    if x.ndim != 3:
        raise ValueError("Input x must have shape (batch, time, channel)")
    if x_sub.ndim != 4 or x_sub.shape[1:] != (1, 1, x_sub.shape[-1]):
        raise ValueError("Input x_sub must have shape (batch, 1, 1, subject_dim)")
    if x.shape[0] != x_sub.shape[0]:
        raise ValueError("Batch dimensions of x and x_sub must match")
    if n_windows <= 0 or n_windows > x.shape[1]:
        raise ValueError("n_windows must be between 1 and the time dimension size")

    B, T, C = x.shape

    # Window size = ceil(T / N)
    win_size = (T + n_windows - 1) // n_windows
    # Stride so windows are evenly spread
    stride = (T - win_size) / max(n_windows - 1, 1)

    windows = []
    for i in range(n_windows):
        start = int(round(i * stride))
        end = start + win_size
        windows.append(x[:, start:end, :].unsqueeze(1))  # (B, 1, window_size, C)

    # Same return logic as slice_time_series
    x_sliced = torch.cat(windows, dim=1).reshape(B * n_windows, -1, C)

    # Repeat subject IDs to align with windows
    x_sub_sliced = x_sub.repeat_interleave(n_windows, dim=0)  # (B * n_windows, 1, 1, subject_dim)

    return x_sliced, x_sub_sliced


class MultiSample:
    def __init__(self, transform, n=2):
        self.transform = transform
        self.num = n

    def __call__(self, x):
        return tuple(self.transform(x) for _ in range(self.num))


def evaluate_old(get_emb_f, ds_name, hyp_c, manifold=None):
    if ds_name != "Inshop":
        emb_head = get_emb_f(ds_type="eval")
        recall_head, recall_all = get_recall_old(*emb_head, *emb_head, ds_name, hyp_c, manifold=manifold)

    else:
        emb_head_query = get_emb_f(ds_type="query")
        emb_head_gal = get_emb_f(ds_type="gallery")
        recall_head, recall_all = get_recall_old(*emb_head_query, *emb_head_gal, ds_name, hyp_c, manifold=manifold)
    return recall_head, recall_all


def calc_recall_at_k(T, Y, k):
    """
    T : [nb_samples] (target labels)
    Y : [nb_samples x k] (k predicted labels/neighbours)
    """

    s = 0
    for t, y in zip(T, Y):
        if t in torch.Tensor(y).long()[:k]:
            s += 1
    return s / (1. * len(T))


def get_recall_old(xq, yq, index_q, xg, yg, index_g, ds_name, hyp_c, manifold=None):
    if ds_name == "SOP":
        k_list = [1, 10, 100, 1000]
    elif ds_name == "Inshop":
        k_list = [1, 10, 20, 30]
    else:
        k_list = [1, 2, 4, 8]

    def part_dist_and_match(xq, yq, index_q, xg, yg, index_g, k):
        if manifold is not None:
            sim = torch.empty(len(xq), len(xg), device="cuda")
            for i in range(len(xq)):
                sim[i: i + 1] = -manifold.sqdist(xq[i: i + 1], xg).unsqueeze(0)  # /manifold.max_dist.pow(2)
        else:
            sim = xq @ xg.T

        sim_diff_idx = torch.where(index_g != index_q.unsqueeze(-1), sim, -torch.ones_like(sim) * 1e7)
        match_counter = ((yq.unsqueeze(-1) == yg[sim_diff_idx.topk(k)[1]]).sum(1) > 0).sum().item()
        return match_counter

    def recall_k(xq, yq, index_q, xg, yg, index_g, k, split_size=5000):
        match_counter = 0
        splits = range(0, len(xq), split_size)
        if split_size < len(xq):
            for i in range(0, len(splits) - 1):
                match_counter += part_dist_and_match(xq[splits[i]:splits[i + 1]], yq[splits[i]:splits[i + 1]],
                                                     index_q[splits[i]:splits[i + 1]], xg, yg, index_g, k)
        match_counter += part_dist_and_match(xq[splits[-1]:], yq[splits[-1]:], index_q[splits[-1]:], xg, yg, index_g, k)
        return match_counter / len(xq)

    recall = [recall_k(xq, yq, index_q, xg, yg, index_g, k) for k in k_list]
    print(recall)
    print(xq.shape)
    return recall[0], recall


def evaluate(get_emb_f, manifold=None):
    emb_head = get_emb_f(ds_type="eval")
    recall_head = get_recall(*emb_head, manifold)
    return recall_head


def get_recall(x, y, manifold=None):
    k_list = [1, 2]

    if manifold is not None:
        dist_m = torch.empty(len(x), len(x), device=x.device)
        for i in range(len(x)):
            dist_m[i: i + 1] = -manifold.sqdist(x[i: i + 1], x)  # /manifold.max_dist.pow(2)
    else:
        dist_m = x @ x.T

    y_cur = y[dist_m.topk(1 + max(k_list), largest=True)[1][:, 1:]]
    y = y.cpu()
    y_cur = y_cur.float().cpu()
    recall = [calc_recall_at_k(y, y_cur, k) for k in k_list]
    print(recall)
    return recall[0], recall


def get_emb(model, dl_eval,skip_head=False):

    model.eval()
    x, y = eval_dataset(model, dl_eval, skip_head)

    model.train()
    return x, y


def eval_dataset(model, dl, t, device):
    all_x, all_y, all_index = [], [], []
    for row in dl:
        xb, yb = row[:-1], row[-1]
        xb = [torch.tensor(t(eeg=x.cpu().numpy())["eeg"]).to(device) for x in xb]
        yb = yb.to(device)
        with torch.no_grad():
            all_x.append(model(*xb, return_embeds=True)[1])
        all_y.append(yb)
    return torch.cat(all_x), torch.cat(all_y)
