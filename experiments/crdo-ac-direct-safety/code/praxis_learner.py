def mlp(in_dim,out_dim,hidden):
    return nn.Sequential(nn.Linear(in_dim,hidden),nn.ReLU(),nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,out_dim))

class SquashedActor(nn.Module):
    def __init__(self,obs_dim,hidden):
        super().__init__(); self.net=mlp(obs_dim,2,hidden)
        nn.init.uniform_(self.net[-1].weight,-.003,.003); nn.init.zeros_(self.net[-1].bias)
    def forward(self,s,deterministic=False):
        params=self.net(s); mean,ls=params[...,:1],params[...,1:].clamp(-5,2)
        u=mean if deterministic else mean+ls.exp()*torch.randn_like(mean)
        a=torch.tanh(u)
        # Stable tanh Jacobian correction. Density is in RAW [-1,1] action units.
        logp=(-.5*((u-mean)/ls.exp()).square()-ls-.5*math.log(2*math.pi)
              -2*(math.log(2)-u-F.softplus(-2*u))).sum(-1,keepdim=True)
        return a,logp

class TwinCritic(nn.Module):
    def __init__(self,obs_dim,hidden,out_dim,ordered=False,nonnegative=False):
        super().__init__(); self.q1=mlp(obs_dim+1,out_dim,hidden); self.q2=mlp(obs_dim+1,out_dim,hidden)
        self.ordered,self.nonnegative=ordered,nonnegative
        for q in (self.q1,self.q2):
            nn.init.uniform_(q[-1].weight,-.003,.003)
            nn.init.constant_(q[-1].bias,-4. if nonnegative else 0.)
    def forward(self,s,a):
        x=torch.cat([s,a],dim=-1); q1,q2=self.q1(x),self.q2(x)
        if self.nonnegative: q1,q2=F.softplus(q1),F.softplus(q2)
        if self.ordered: q1,q2=q1.sort(-1).values,q2.sort(-1).values
        return q1,q2

def quantile_huber(pred,target,kappa=1.):
    # pred: B,K ; target: B,K'. Targets are detached by the caller.
    k=pred.shape[-1]; tau=(torch.arange(k,device=pred.device,dtype=pred.dtype)+.5)/k
    delta=target.unsqueeze(1)-pred.unsqueeze(2)
    huber=torch.where(delta.abs()<=kappa,.5*delta.square(),kappa*(delta.abs()-.5*kappa))
    return ((tau[None,:,None]-(delta.detach()<0).to(pred.dtype)).abs()*huber/kappa).mean()

def lower_tail(q,alpha):
    # Fractional-bin integration avoids incorrectly rounding 5% of 64 to 3 or 4 atoms.
    k=q.shape[-1]; left=torch.arange(k,device=q.device,dtype=q.dtype)/k
    weights=(torch.minimum(left+1/k,torch.tensor(alpha,device=q.device))-left).clamp(min=0)/alpha
    return (q*weights).sum(-1,keepdim=True)

class ActorCriticLearner:
    def __init__(self,obs_dim,cfg,variant,device):
        self.cfg,self.variant,self.device=cfg,variant,device
        self.distributional=variant.startswith("CRDO")
        self.conservative=self.distributional and variant!="CRDO-no-CQL"
        self.tail=self.distributional and variant!="CRDO-no-tail"
        self.constrained=self.distributional and variant!="CRDO-no-CMDP"
        self.actor=SquashedActor(obs_dim,cfg.hidden).to(device)
        k=cfg.quantiles if self.distributional else 1
        self.critics=TwinCritic(obs_dim,cfg.hidden,k,ordered=self.distributional).to(device)
        self.targets=copy.deepcopy(self.critics).requires_grad_(False)
        self.costs=TwinCritic(obs_dim,cfg.hidden,2,nonnegative=True).to(device) if self.constrained else None
        self.cost_targets=copy.deepcopy(self.costs).requires_grad_(False) if self.costs is not None else None
        self.log_entropy=nn.Parameter(torch.tensor(math.log(cfg.initial_entropy),device=device))
        self.duals=torch.zeros(2,device=device)
        self.budgets=torch.tensor([cfg.turnover_budget,cfg.drawdown_excess_budget],device=device)
        self.actor_opt=torch.optim.Adam(self.actor.parameters(),lr=cfg.lr)
        self.critic_opt=torch.optim.Adam(self.critics.parameters(),lr=cfg.lr)
        self.entropy_opt=torch.optim.Adam([self.log_entropy],lr=cfg.lr)
        self.cost_opt=torch.optim.Adam(self.costs.parameters(),lr=cfg.lr) if self.costs else None
        self.updates=0

    def update(self,batch,initial_obs=None):
        cfg=self.cfg; s,a,r,c,sn,d=[batch[k] for k in ("obs","action","reward","cost","next_obs","done")]
        alpha=self.log_entropy.exp().detach()
        with torch.no_grad():
            an,lpn=self.actor(sn); z1,z2=self.targets(sn,an)
            if self.distributional:
                # Physical scaled net-return distribution, NOT entropy-augmented soft returns.
                choose=(z1.mean(-1,keepdim=True)<=z2.mean(-1,keepdim=True))
                zn=torch.where(choose,z1,z2)
                target=r+cfg.gamma*(1-d)*zn
            else:
                # Standard SAC-v2 soft Bellman target, including future entropy.
                target=r+cfg.gamma*(1-d)*(torch.minimum(z1,z2)-alpha*lpn)
        q1,q2=self.critics(s,a)
        if self.distributional: bellman=quantile_huber(q1,target)+quantile_huber(q2,target)
        else: bellman=F.mse_loss(q1,target)+F.mse_loss(q2,target)
        cql=torch.tensor(0.,device=self.device)
        if self.conservative:
            b=len(s); k=cfg.cql_samples
            uniform=torch.rand((b,k,1),device=self.device)*2-1
            sr=s[:,None,:].expand(-1,k,-1).reshape(b*k,-1)
            u1,u2=self.critics(sr,uniform.reshape(-1,1))
            for u,q in ((u1,q1),(u2,q2)):
                uq=u.mean(-1).reshape(b,k)
                # Monte-Carlo log integral over [-1,1]; proposal density is 1/2.
                lse=cfg.cql_temperature*(torch.logsumexp(uq/cfg.cql_temperature,dim=1)-math.log(k)+math.log(2))
                cql=cql+(lse-q.mean(-1)).mean()
        critic_loss=bellman+cfg.cql_weight*cql
        self._step(critic_loss,self.critic_opt,self.critics.parameters())
        cost_loss=torch.tensor(0.,device=self.device)
        if self.constrained:
            with torch.no_grad():
                cn1,cn2=self.cost_targets(sn,an)
                ct=(1-cfg.gamma)*c+cfg.gamma*(1-d)*torch.maximum(cn1,cn2)
            c1,c2=self.costs(s,a); cost_loss=F.mse_loss(c1,ct)+F.mse_loss(c2,ct)
            self._step(cost_loss,self.cost_opt,self.costs.parameters())
        # Critic parameters are frozen for actor differentiation, NOT the action gradient.
        self.critics.requires_grad_(False)
        if self.costs: self.costs.requires_grad_(False)
        ap,lp=self.actor(s); p1,p2=self.critics(s,ap)
        beta=cfg.tail_weight if self.tail else 0.
        if self.distributional:
            value=torch.minimum(p1.mean(-1,keepdim=True)+beta*lower_tail(p1,cfg.tail_alpha),
                                p2.mean(-1,keepdim=True)+beta*lower_tail(p2,cfg.tail_alpha))
        else: value=torch.minimum(p1,p2)
        penalty=torch.zeros_like(value)
        if self.constrained:
            pc1,pc2=self.costs(s,ap)
            penalty=((torch.maximum(pc1,pc2)-self.budgets)*self.duals.detach()).sum(-1,keepdim=True)
        actor_loss=(alpha*lp-value+penalty).mean()
        self._step(actor_loss,self.actor_opt,self.actor.parameters())
        self.critics.requires_grad_(True)
        if self.costs: self.costs.requires_grad_(True)
        entropy_loss=-(self.log_entropy*(lp.detach()+cfg.target_entropy)).mean()
        self._step(entropy_loss,self.entropy_opt,[self.log_entropy])
        with torch.no_grad():
            self.log_entropy.clamp_(-10,2)
            if self.constrained:
                si=initial_obs if initial_obs is not None else s
                ai,_=self.actor(si); ci1,ci2=self.costs(si,ai)
                gap=torch.maximum(ci1,ci2).mean(0)-self.budgets
                self.duals.add_(cfg.dual_lr*gap).clamp_(0,cfg.dual_max)
            for online,target_net in ((self.critics,self.targets),(self.costs,self.cost_targets)):
                if online is not None:
                    for p,tp in zip(online.parameters(),target_net.parameters()): tp.lerp_(p,cfg.tau)
        self.updates+=1
        # Read GPU scalars only at log intervals, not every gradient step.
        if self.updates%cfg.log_every==0 or self.updates==1:
            return {"update":self.updates,"bellman_loss":float(bellman.detach()),
                "cql_penalty":float(cql.detach()),"actor_loss":float(actor_loss.detach()),
                "cost_critic_loss":float(cost_loss.detach()),"entropy_temperature":float(self.log_entropy.exp().detach()),
                "lambda_turnover":float(self.duals[0]),"lambda_drawdown":float(self.duals[1])}
        return None

    @staticmethod
    def _step(loss,opt,parameters):
        # error_if_nonfinite fails fast on NaN gradients, rather than caching a corrupt model.
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(list(parameters),10.,error_if_nonfinite=True); opt.step()

    def state(self):
        return {"actor":self.actor.state_dict(),"critics":self.critics.state_dict(),"targets":self.targets.state_dict(),
                "costs":self.costs.state_dict() if self.costs else None,
                "cost_targets":self.cost_targets.state_dict() if self.cost_targets else None,
                "actor_opt":self.actor_opt.state_dict(),"critic_opt":self.critic_opt.state_dict(),
                "cost_opt":self.cost_opt.state_dict() if self.cost_opt else None,
                "entropy_opt":self.entropy_opt.state_dict(),"log_entropy":self.log_entropy.detach(),
                "duals":self.duals,"updates":self.updates}

    def load(self,state):
        for name in ("actor","critics","targets","costs","cost_targets"):
            module=getattr(self,name)
            if module is not None: module.load_state_dict(state[name])
        with torch.no_grad(): self.log_entropy.copy_(state["log_entropy"]); self.duals.copy_(state["duals"])
        for name in ("actor_opt","critic_opt","cost_opt","entropy_opt"):
            opt=getattr(self,name)
            if opt is not None: opt.load_state_dict(state[name])
        self.updates=int(state["updates"])

    def inference_actor(self):
        return copy.deepcopy(self.actor).to("cpu").eval().requires_grad_(False)
