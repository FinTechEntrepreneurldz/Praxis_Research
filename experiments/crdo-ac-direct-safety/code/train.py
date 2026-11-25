import copy,datetime,hashlib,json,math,os,sys,time
import numpy as np,torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from types import SimpleNamespace
from tasks import ROOT,verify,digest
from env import ledger,lookup
exec(compile((ROOT/'code'/'praxis_learner.py').read_text(),'praxis_learner.py','exec'))
SEEDS=(42,123,2026)
MODES={'sac':'SAC','crdo':'CRDO-AC','no_tail':'CRDO-no-tail'}
STEPS=20000;BATCH=32
CFG=dict(hidden=256,quantiles=64,gamma=.995,tau=.005,lr=.0003,cql_samples=10,cql_temperature=1.,cql_weight=.1,tail_weight=.5,tail_alpha=.05,initial_entropy=.05,target_entropy=-1.,dual_lr=.05,dual_max=100.,turnover_budget=.16,drawdown_excess_budget=.40,log_every=1000)
def arrhash(x):return hashlib.sha256(x.tobytes()).hexdigest()
def start_model(seed,mode,n):
 torch.manual_seed(seed);return ActorCriticLearner(n,SimpleNamespace(**CFG),MODES[mode],torch.device('cpu'))
def ahash(m):return hashlib.sha256(b''.join(t.detach().cpu().numpy().tobytes() for t in m.actor.state_dict().values())).hexdigest()
def train(seed,mode):
 verify();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 data=lookup('train');buf,init=ledger(data);rand=np.random.default_rng(11000+seed)
 ix=rand.integers(len(buf['obs']),size=(STEPS,BATCH),dtype='int32');ii=rand.integers(len(init),size=(STEPS,BATCH),dtype='int32')
 bh=hashlib.sha256(''.join(arrhash(x) for x in [*buf.values(),init]).encode()).hexdigest();mh=arrhash(ix)+':'+arrhash(ii)
 buf={k:torch.from_numpy(v) for k,v in buf.items()};init=torch.from_numpy(init)
 m=start_model(seed,mode,init.shape[-1]);initial=ahash(m)
 folder=ROOT/'models';folder.mkdir(exist_ok=True);final=folder/f'{mode}_{seed}.pt';cp=folder/f'{mode}_{seed}.resume';log=folder/f'{mode}_{seed}.json'
 if final.exists() and log.exists():
  a=json.loads(log.read_text());assert a['updates']==STEPS and a['sha256']==digest(final) and a['buffer']==bh and a['minibatches']==mh;return
 metrics=[];elapsed=0.
 if cp.exists():
  saved=torch.load(cp,map_location='cpu',weights_only=True)
  assert saved['buffer']==bh and saved['minibatches']==mh
  m.load(saved['model']);torch.set_rng_state(saved['rng']);metrics=saved['traces'];elapsed=saved['elapsed']
 start=time.monotonic()
 for step in range(m.updates,STEPS):
  metrics_row=m.update({k:v[ix[step]] for k,v in buf.items()},init[ii[step]])
  if metrics_row:
   assert all(math.isfinite(v) for v in metrics_row.values())
   metrics.append(metrics_row)
  if (step+1)%2000==0:
   tmp=cp.with_suffix('.part');torch.save(dict(model=m.state(),rng=torch.get_rng_state(),buffer=bh,minibatches=mh,traces=metrics,elapsed=elapsed+time.monotonic()-start),tmp);tmp.replace(cp)
   print(mode,seed,step+1,round(elapsed+time.monotonic()-start),flush=True)
 assert m.updates==STEPS and ahash(m)!=initial
 tmp=final.with_suffix('.tmp');torch.save(dict(actor=m.actor.state_dict(),critics=m.critics.state_dict(),costs=m.costs.state_dict() if m.costs else None,duals=m.duals,config=CFG,updates=STEPS,mode=mode,seed=seed),tmp);tmp.replace(final)
 log.write_text(json.dumps(dict(seed=seed,mode=mode,updates=m.updates,seconds=elapsed+time.monotonic()-start,buffer=bh,minibatches=mh,rows=len(buf['obs']),batch_size=BATCH,
     init_actor=initial,final_actor=ahash(m),sha256=digest(final),traces=metrics,finished=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2))
 cp.unlink(missing_ok=True)
if __name__=='__main__':train(int(sys.argv[1]),sys.argv[2])
