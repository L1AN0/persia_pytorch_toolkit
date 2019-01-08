import torch

class RiemannAUCMeter():
    """
    Efficient PyTorch AUC Estimator.
    """
    def __init__(self, num_bins=100000):
        self.num_bins = num_bins
        self.p_cnt = torch.zeros(self.num_bins, dtype=torch.long)
        self.n_cnt = torch.zeros(self.num_bins, dtype=torch.long)

    def add(self, outputs, labels):
        """Outputs should be probabilities from 0 to 1."""
        indices = torch.clamp(outputs * self.num_bins, min=0, max=self.num_bins - 1).long()
        p_indices = torch.masked_select(indices, labels != 0)
        n_indices = torch.masked_select(indices, labels == 0)
        p_mask = torch.sparse.FloatTensor(p_indices.unsqueeze(0), torch.ones_like(p_indices), torch.Size([self.num_bins]))
        n_mask = torch.sparse.FloatTensor(n_indices.unsqueeze(0), torch.ones_like(n_indices), torch.Size([self.num_bins]))
        self.p_cnt += p_mask
        self.n_cnt += n_mask

    def value(self):
        p_sum = self.p_cnt.sum().item()
        n_sum = self.n_cnt.sum().item()

        prod = torch.dot(self.p_cnt, self.n_cnt).item() * 0.5
        n_cumsum = self.n_cnt.cumsum(0)
        up_sum = torch.dot(self.p_cnt, n_cumsum).item() + prod
        try:
            return float(up_sum)/float(p_sum*n_sum)
        except ZeroDivisionError:
            return 0


if __name__ == "__main__":
    meter = RiemannAUCMeter()
    torch.manual_seed(7)
    outputs = torch.rand(10000)
    labels = torch.rand(10000) > 0.4
    for _ in range(100):
        meter.add(outputs, labels.float())
        print(meter.value())
