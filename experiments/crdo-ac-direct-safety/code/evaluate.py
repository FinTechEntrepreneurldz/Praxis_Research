import datetime,itertools,json,numpy as np,pandas as pd,torch
from tasks import ROOT,digest,verify
from env import lookup,state,action,QUOTA,ARMS
from train import SEEDS,MODES,STEPS,start_model
def evaluate():
 verify();p=ROOT/'models';rdir=ROOT/'results';rdir.mkdir(exist_ok=True);torch.set_num_threads(1)
 meta=[json.loads((p/f'{m}_{s}.json').read_text()) for s in SEEDS for m in MODES]
 assert all(x['updates']==STEPS and x['sha256']==digest(p/('%s_%s.pt'%(x['mode'],x['seed']))) for x in meta)
 for s in SEEDS:
  for k in ['buffer','minibatches','init_actor']:assert len({v[k] for v in meta if v['seed']==s})==1
 audit={v['mode']+'_'+str(v['seed']):v['sha256'] for v in meta}
 (rdir/'evaluation_lock.json').write_text(json.dumps(audit,indent=2))
 test=lookup('test');rows=[];controls=[]
 for seed in SEEDS:
  for mode in MODES:
   ck=torch.load(p/f'{mode}_{seed}.pt',map_location='cpu',weights_only=True);m=start_model(seed,mode,len(state(test[0],QUOTA)));m.actor.load_state_dict(ck['actor']);m.actor.eval()
   for i in range(0,len(test),4):
    rem=np.array(QUOTA)
    for rec in test[i:i+4]:
     x=torch.from_numpy(state(rec,rem)).unsqueeze(0)
     with torch.inference_mode():value=float(m.actor(x,deterministic=True)[0].item())
     a=action(value,rem);rem[a]-=1;t=rec['task']
     rows.append(dict(domain=t['domain'],queue=t['queue'],id=t['id'],seed=seed,mode=mode,decision=ARMS[a],continuous=value,
        draft=rec['acts'][0],final=rec['acts'][a],cost=(0,1,3)[a],**rec['grades'][a]))
    assert not rem.any()
 for rec in test:
  for a in range(3):
   controls.append(dict(id=rec['task']['id'],domain=rec['task']['domain'],queue=rec['task']['queue'],decision=ARMS[a],final=rec['acts'][a],**rec['grades'][a]))
 frame=pd.DataFrame(rows);frame.to_csv(rdir/'decisions.csv',index=False);pd.DataFrame(controls).to_csv(rdir/'branch_outcomes.csv',index=False)
 assert frame.groupby(['mode','seed','queue']).cost.sum().eq(4).all()
 return frame
def compare(frame,chall='crdo',base='sac',domain='pooled'):
 f=frame if domain=='pooled' else frame[frame.domain==domain]
 def sums(md):return f[f['mode']==md].groupby(['domain','queue','seed'])[['severe','completed','eligible']].sum().groupby(['domain','queue']).mean()
 a=sums(chall);b=sums(base);assert a.index.equals(b.index) and (a.eligible==b.eligible).all()
 arr=np.c_[a.severe-b.severe,a.completed-b.completed,a.eligible];n=len(arr);groups=[np.flatnonzero(a.index.get_level_values(0)==d) for d in sorted(set(a.index.get_level_values(0)))]
 def effects(x):return np.stack([x[...,0].sum(-1)/(x.shape[-2]*4),x[...,1].sum(-1)/x[...,2].sum(-1)],-1)
 rng=np.random.default_rng(88912);boot=[]
 for _ in range(100):
  ix=np.concatenate([rng.choice(g,size=(200,len(g)),replace=True) for g in groups],axis=1);boot.extend(effects(arr[ix]))
 ci=np.quantile(boot,[.025,.05,.95,.975],axis=0);delta=effects(arr)
 return dict(domain=domain,challenger=chall,baseline=base,n=n*4,eligible=int(a.eligible.sum()),severe_delta=float(delta[0]),completion_delta=float(delta[1]),severe_upper=float(ci[2,0]),completion_lower=float(ci[1,1]),
   severe_ci=ci[[0,3],0].tolist(),completion_ci=ci[[0,3],1].tolist(),passed=bool(delta[0]<0 and delta[1]>=0 and ci[2,0]<0 and ci[1,1]>=0))
def report(frame):
 r=ROOT/'results';out=[compare(frame,'crdo',base,d) for base in ['sac','no_tail'] for d in ['pooled','archive','equipment']]
 (r/'paired.json').write_text(json.dumps(out,indent=2))
 table=frame.groupby(['domain','mode','seed'])[['severe','completed','eligible','correct','cost']].sum()
 total=frame.groupby(['mode','seed'])[['severe','completed','eligible','correct','cost']].sum();total.to_csv(r/'pooled_by_seed.csv');table.to_csv(r/'domain_by_seed.csv')
 print(total.to_string(),flush=True);print(json.dumps(out[0]),flush=True);print(json.dumps(out[3]),flush=True)
if __name__=='__main__':report(evaluate())
