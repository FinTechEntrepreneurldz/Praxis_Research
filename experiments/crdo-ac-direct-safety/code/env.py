import itertools,json,numpy as np
from tasks import ROOT,cases,grade,outcomes,CHOICES,digest
ARMS=('release','revise','escalate')
QUOTA=(2,1,1)
def lookup(split):
 tasks=cases(split);maps=[]
 for stage in ('draft','revision','escalation'):
  p=ROOT/'raw'/f'{split}_{stage}.jsonl';md=json.loads(p.with_suffix('.meta.json').read_text());assert md['records']==len(tasks) and md['sha256']==digest(p)
  items=[json.loads(x) for x in p.read_text().splitlines()];assert [r['id'] for r in tasks]==[i['id'] for i in items]
  assert all(row['menu'][np.argmax(it['logits'])]==it['action'] for row,it in zip(tasks,items))
  maps.append(items)
 recs=[]
 for i,r in enumerate(tasks):
  d=maps[0][i];v=np.array(d['logits']);v=np.exp(v-v.max());probs=v/v.sum();order=np.array([r['menu'].index(c) for c in CHOICES])
  ob=np.r_[r['flags'],probs[order],np.eye(4)[CHOICES.index(d['action'])],np.log1p(np.sort(d['logits'])[-1]-np.sort(d['logits'])[-2])/3]
  recs.append(dict(task=r,public=ob,acts=[m[i]['action'] for m in maps],grades=[outcomes(r,m[i]['action']) for m in maps]))
 return recs
def state(rec,remaining):
 return np.r_[rec['public'],np.array(remaining)/np.array(QUOTA),sum(remaining)/4].astype('float32')
def action(value,remaining):
 # Order the remaining dispositions; every legal disposition gets an equal interval on [-1,1].
 usable=np.flatnonzero(remaining)
 assert len(usable) and -1.000001<=value<=1.000001
 return int(usable[min(len(usable)-1,int((np.clip(value,-1,1)+1)*.5*len(usable)))])
def ledger(recs):
 assert {r['task']['domain'] for r in recs}=={'parcel','admission'}
 schedules=sorted(set(itertools.permutations((0,0,1,2))));rng=np.random.default_rng(62218)
 rows=[];starts=[]
 for begin in range(0,len(recs),4):
  queue=recs[begin:begin+4];starts.append(state(queue[0],QUOTA))
  for schedule in schedules:
   remaining=np.array(QUOTA)
   for pos,(rec,chosen) in enumerate(zip(queue,schedule)):
    s=state(rec,remaining);usable=np.flatnonzero(remaining);rank=usable.tolist().index(chosen)
    remaining[chosen]-=1
    ns=state(queue[pos+1],remaining) if pos<3 else np.zeros_like(s)
    g=rec['grades'][chosen]
    for j in range(3):
     raw=-1+2*(rank+(j+rng.random())/3)/len(usable);prev=remaining.copy();prev[chosen]+=1
     assert action(raw,prev)==chosen
     rows.append((s,[raw],[g['reward']],[g['severe']/.005,g['missed']/.005],ns,[int(pos==3)]))
 return {k:np.asarray(v,dtype=np.float32) for k,v in zip(('obs','action','reward','cost','next_obs','done'),zip(*rows))},np.array(starts)
