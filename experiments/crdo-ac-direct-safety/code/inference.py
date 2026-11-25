"""Produce full-information LLM response ledgers, with review outcomes hidden from the actor."""
import argparse,json,time,os,datetime,hashlib
from pathlib import Path
from tasks import ROOT,D,cases,prompt,verify,digest
from fetch_models import MODELS
def run(split,stage,batch=16):
 import torch,transformers
 from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache
 verify()
 tag='qwen' if stage=='escalation' else 'smol';repo,rev,folder=MODELS[tag]
 modeldir=Path(os.environ.get('DECISION_MODEL_DIR',ROOT/'assets'))/folder
 assert (modeldir/'revision.txt').read_text()==rev
 dev='cuda' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'cpu'
 torch.set_num_threads(int(os.environ.get('DECISION_THREADS','4')));torch.manual_seed(98217)
 tok=AutoTokenizer.from_pretrained(modeldir,local_files_only=True,trust_remote_code=False)
 model=AutoModelForCausalLM.from_pretrained(modeldir,torch_dtype=torch.bfloat16,attn_implementation='eager',local_files_only=True,trust_remote_code=False).eval().to(dev)
 lbl=[tok.encode(s,add_special_tokens=False) for s in 'ABCD'];assert all(len(x)==1 for x in lbl)
 ids4=torch.tensor([x[0] for x in lbl],device=dev);head=model.get_output_embeddings()
 W=head.weight.index_select(0,ids4);bias=None if head.bias is None else head.bias.index_select(0,ids4)
 out=ROOT/'raw';out.mkdir(exist_ok=True);path=out/f'{split}_{stage}.jsonl';rows=cases(split)
 drafts={}
 if stage=='revision':drafts={x['id']:x['action'] for x in map(json.loads,(out/f'{split}_draft.jsonl').read_text().splitlines())};assert len(drafts)==len(rows)
 old=[json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
 assert [x['id'] for x in old]==[r['id'] for r in rows[:len(old)]]
 start=time.monotonic();checks=[]
 with torch.inference_mode(),path.open('a') as log:
  for domain in dict.fromkeys(r['domain'] for r in rows):
   group=[r for r in rows if r['domain']==domain]
   text=[tok.apply_chat_template(prompt(r,stage,drafts.get(r['id'])),tokenize=False,add_generation_prompt=True) for r in group]
   toks=[tok.encode(x,add_special_tokens=False) for x in text];n=0
   while n<min(map(len,toks)) and len({x[n] for x in toks})==1:n+=1
   assert n>10
   common=model.base_model(input_ids=torch.tensor([toks[0][:n]],device=dev),use_cache=True).past_key_values.to_legacy_cache()
   offset=rows.index(group[0]);skip=max(0,min(len(old)-offset,len(group)))
   for i in range(skip,len(group),batch):
    local=list(range(i,min(i+batch,len(group))));ss=[toks[k][n:] for k in local]
    lengths=[len(u) for u in ss];m=max(lengths);b=len(local);pad=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    x=torch.tensor([u+[pad]*(m-len(u)) for u in ss],device=dev)
    mask=torch.cat([torch.ones(b,n,device=dev,dtype=torch.bool),torch.arange(m,device=dev)[None,:]<torch.tensor(lengths,device=dev)[:,None]],1)
    cache=DynamicCache.from_legacy_cache(tuple((kk.expand(b,-1,-1,-1),vv.expand(b,-1,-1,-1)) for kk,vv in common))
    hidden=model.base_model(input_ids=x,attention_mask=mask,past_key_values=cache,use_cache=True).last_hidden_state[torch.arange(b,device=dev),torch.tensor(lengths,device=dev)-1]
    scores=torch.nn.functional.linear(hidden,W,bias).float().cpu()
    assert torch.isfinite(scores).all()
    if i==0:
     h=model.base_model(input_ids=torch.tensor([toks[0]],device=dev),use_cache=False).last_hidden_state[:,-1]
     raw=torch.nn.functional.linear(h,W,bias).float().cpu()[0]
     diff=float((raw-scores[0]).abs().max());assert diff<=.5
     checks.append(dict(domain=domain,max_raw_logit_delta=diff,top_equal=bool(raw.argmax()==scores[0].argmax())))
    for j,k in enumerate(local):
     r=group[k];v=scores[j].tolist();a=int(scores[j].argmax());rec=dict(id=r['id'],split=split,stage=stage,model=repo,revision=rev,action=r['menu'][a],letter='ABCD'[a],logits=v,
       prompt=text[k],input_tokens=len(toks[k]),utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
     log.write(json.dumps(rec,separators=(',',':'))+'\n')
    log.flush()
   print(split,stage,domain,'complete',round(time.monotonic()-start,1),flush=True)
 assert len(path.read_text().splitlines())==len(rows)
 (out/f'{split}_{stage}.meta.json').write_text(json.dumps(dict(records=len(rows),seconds=time.monotonic()-start,sha256=digest(path),model=repo,commit=rev,device=dev,dtype='bfloat16',torch=torch.__version__,transformers=transformers.__version__,checks=checks,model_hashes=json.loads((modeldir/'hashes.json').read_text())),indent=2))
if __name__=='__main__':
 import sys
 run(*sys.argv[1:3])
