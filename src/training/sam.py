"""SAM (Sharpness-Aware Minimization) optimizer wrapper.
  Pass 1 — perturb weights toward the local worst-case neighbor:
      e_hat = rho * grad / ||grad||
      param += e_hat

  Pass 2 — compute gradient at the perturbed point, restore original weights,
            then let the base optimizer take its normal update step:
      param -= e_hat
      base_optimizer.step()
"""

from __future__ import annotations

import torch


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer: type, rho: float = 0.05, **kwargs):
        assert rho >= 0, f"rho must be non-negative, got {rho}"
        defaults = {"rho": rho, **kwargs}
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_hat = p.grad * scale
                p.add_(e_hat)                        
                self.state[p]["e_hat"] = e_hat       
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_hat"])       
        self.base_optimizer.step()                   
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self) -> torch.Tensor:
        device = next(
            p for group in self.param_groups for p in group["params"] if p.grad is not None
        ).device
        norms = [
            p.grad.norm(2).to(device)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        return torch.stack(norms).norm(2)

    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

    def step(self, closure=None):
        raise NotImplementedError("SAM requires explicit first_step / second_step calls.")
