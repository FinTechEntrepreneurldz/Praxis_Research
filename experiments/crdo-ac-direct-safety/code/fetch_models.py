"""Fetch only the inference assets at the two immutable public model revisions."""
from pathlib import Path
import sys,json,urllib.request,concurrent.futures,shutil,hashlib,time
BASE=Path(__file__).resolve().parents[1]
MODELS={'smol':('HuggingFaceTB/SmolLM2-360M-Instruct','a10cc1512eabd3dde888204e902eca88bddb4951','model'),'qwen':('Qwen/Qwen2.5-1.5B-Instruct','989aa7980e4cf806f80c7fef2b1adb7bc71aa306','qwen')}
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(2**21),b''):h.update(chunk)
 return h.hexdigest()
def fetch(tag):
 repo,rev,folder=MODELS[tag];root=BASE/'assets'/folder;root.mkdir(parents=True,exist_ok=True)
 url='https://huggingface.co/api/models/'+repo+'/revision/'+rev+'?blobs=true'
 d=json.load(urllib.request.urlopen(url,timeout=40));assert d['sha']==rev
 valid={'model.safetensors','tokenizer.json','tokenizer_config.json','special_tokens_map.json','config.json','generation_config.json','vocab.json','merges.txt','chat_template.jinja','model.safetensors.index.json'}
 chosen=[x for x in d['siblings'] if x['rfilename'] in valid or x['rfilename'].endswith('.safetensors')]
 def do(x):
  name=x['rfilename'];f=root/name
  for k in range(4):
   try:
    if not f.exists() or f.stat().st_size!=x.get('size',f.stat().st_size):
     u=f'https://huggingface.co/{repo}/resolve/{rev}/{name}'
     with urllib.request.urlopen(u,timeout=90) as resp,(root/(name+'.part')).open('wb') as tmp:shutil.copyfileobj(resp,tmp,2**20)
     (root/(name+'.part')).replace(f)
    sha=digest(f)
    if x.get('lfs'):assert sha==x['lfs']['sha256']
    print(tag,name,f.stat().st_size,flush=True);return name,sha
   except Exception:
    if k==3:raise
    time.sleep(2**k)
 with concurrent.futures.ThreadPoolExecutor(5) as ex: hashes=dict(ex.map(do,chosen))
 (root/'revision.txt').write_text(rev);(root/'hashes.json').write_text(json.dumps(hashes,indent=2))
if __name__=='__main__':fetch(sys.argv[1])
