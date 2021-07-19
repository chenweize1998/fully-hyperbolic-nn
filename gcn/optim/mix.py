from apex.optimizers import FusedAdam
from geoopt.manifolds.base import Manifold
from optim import RiemannianAdam
from geoopt import ManifoldParameter


class MixAdam():
    def __init__(self,
                 params,
                 lr=1e-3,
                 bias_correction=True,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 adam_w_mode=True,
                 weight_decay=0.,
                 amsgrad=False,
                 set_grad_none=True,
                 stabilize=10,
                 max_grad_norm=1.):
        rparams = []
        eparams = []
        for r in params:
            if r.requires_grad:
                if isinstance(r, ManifoldParameter):
                    rparams.append(r)
                else:
                    eparams.append(r)
        self.op1 = RiemannianAdam(rparams,
                                  lr=lr,
                                  betas=betas,
                                  eps=eps,
                                  weight_decay=weight_decay,
                                  stabilize=stabilize,
                                  max_grad_norm=max_grad_norm)
        self.op2 = FusedAdam(eparams,
                             lr=lr,
                             bias_correction=bias_correction,
                             betas=(0.9, 0.999),
                             eps=1e-8,
                             adam_w_mode=adam_w_mode,
                             weight_decay=0.,
                             amsgrad=amsgrad,
                             set_grad_none=set_grad_none)
        self.max_grad_norm = max_grad_norm

    def step(self):
        pass