import torch
from torch.optim import AdamW


class FP32StateAdamW(AdamW):
    """
    AdamW that keeps its states in float32 for low-precision (bf16 / fp16) weights.

    torch.optim.AdamW allocates its states in the dtype of the weights, and float16 cannot hold the
    second moment (g^2 underflows to 0). Here every step is taken in float32 instead: the weights and
    their gradients are cast to float32, the regular AdamW step is applied, and the updated weights
    are rounded back to their own dtype.
    """
    @torch.no_grad()
    def step(self, closure=None):
        params = [p for group in self.param_groups for p in group['params'] if p.grad is not None]
        dtypes = [p.dtype for p in params]

        for p in params:
            p.data, p.grad.data = p.data.float(), p.grad.data.float()

        loss = super().step(closure)

        for p, dtype in zip(params, dtypes):
            p.data, p.grad.data = p.data.to(dtype), p.grad.data.to(dtype)

        return loss
