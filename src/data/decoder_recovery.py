"""Validate optional same-source decoder recovery evidence; never retry media."""
import ast,hashlib,json,re,math
from pathlib import Path

HEADER_PROBE_EXECUTABLE_SHA256='2ce797a0f88d7f067180338fb227f7b1928ea727bd9a4d7a1d022f7c52af71a3'
FILES={'parent_preflight_sha256':'parent-preflight.json','parent_rows_sha256':'parent-rows.jsonl','audit_sha256':'attempts.jsonl','helper_sha256':'recover.py'}
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def journal(path):
 raw=Path(path).read_bytes()
 if raw and not raw.endswith(b'\n'):raise ValueError('Incomplete recovery journal')
 return [json.loads(line) for line in raw.splitlines() if line.strip()]
def transient(row):
 error=row.get('error','')
 return row.get('status')=='invalid' and row.get('reason')!='annotation_geometry_conflict' and (error.startswith('PermissionError:') or re.search(r'\[WinError 32\]',error) is not None)
def function_hash(path):
 text=Path(path).read_text(encoding='utf-8');node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='inspect')
 return hashlib.sha256(ast.get_source_segment(text,node).encode()).hexdigest()
def successful(row):
 n=row.get('num_segments');d=row.get('duration');f=row.get('fps')
 if type(n) is not int or not 1<=n<=6 or not all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in (d,f)) or d<=(n-1)*5 or row.get('last_frame_decoded') is not True:return False
 return type(row.get('last_frame')) is int and row['last_frame']==math.floor(((n-1)*5+(min(n*5,d)-(n-1)*5)*.75)*f)
def structural_header_error(text):
 if not isinstance(text,str):return False
 required=('STSC','STCO','error reading header','Invalid data found when processing input')
 forbidden=('PermissionError','WinError','Cannot allocate memory','Resource temporarily unavailable','Permission denied','No such file','out of memory')
 return all(token in text for token in required) and not any(token in text for token in forbidden)
def checked_probe(probe,row,receipt):
 if not isinstance(probe,dict) or set(probe)!={'receipt_sha256','receipt_json'} or not isinstance(probe['receipt_json'],str) or hashlib.sha256(probe['receipt_json'].encode('utf-8')).hexdigest()!=probe['receipt_sha256']:raise ValueError('Invalid container probe receipt hash')
 p=json.loads(probe['receipt_json'])
 fields={'command','raw_sha256','source_bytes','ffmpeg_executable','ffmpeg_executable_sha256','ffmpeg_version_stdout','ffmpeg_version_stderr','exit_code','stdout','stderr','source_unchanged_after','media_rewritten','probe'}
 if not isinstance(p,dict) or set(p)!=fields:raise ValueError('Invalid container probe schema')
 command=p['command']
 if not isinstance(command,list) or len(command)!=12 or command[0]!=p['ffmpeg_executable'] or command[1:4]!=['-v','error','-i'] or command[5:]!=['-map','0:v:0','-frames:v','1','-f','null','-'] or not isinstance(command[4],str) or command[4].replace('\\','/').split('/')[-1]!=row['video']:raise ValueError('Invalid container probe command')
 if p['raw_sha256']!=row['sha256'] or type(p['source_bytes']) is not int or p['source_bytes']!=receipt['bytes'] or p['source_bytes']<=0 or not isinstance(p['ffmpeg_executable_sha256'],str) or p['ffmpeg_executable_sha256']!=HEADER_PROBE_EXECUTABLE_SHA256 or not isinstance(p['ffmpeg_version_stdout'],str) or not p['ffmpeg_version_stdout'].startswith('ffmpeg version ') or type(p['exit_code']) is not int or p['exit_code']==0 or p['source_unchanged_after'] is not True or p['media_rewritten'] is not False or p['probe']!='first_video_frame_to_null' or not structural_header_error(p['stderr']):raise ValueError('Container probe does not confirm same-source header rejection')
 return p
def checked_recovery(root,decoded,receipt_sha,source):
 """Return fixed proof artifacts only after full parent/audit/replacement checks."""
 recovery=decoded.get('recovery')
 if recovery is None:return []
 expected=set(FILES)|{'unchanged_inspect_source_sha256','inspect_function_sha256','replacements','targets_used_for_filtering'}
 if not isinstance(recovery,dict) or set(recovery)!=expected or recovery['targets_used_for_filtering'] is not False:raise ValueError('Invalid decoder recovery schema')
 base=(Path(root)/'decoder/recovery').resolve();root=Path(root).resolve()
 if not base.is_relative_to(root):raise ValueError('Recovery path escapes proof')
 paths=[]
 for field,name in FILES.items():
  path=(base/name).resolve()
  if not path.is_relative_to(base) or digest(path)!=recovery[field]:raise ValueError('Changed decoder recovery artifact')
  paths.append(path)
 source_sha=digest(source);function_sha=function_hash(source)
 if recovery['unchanged_inspect_source_sha256']!=source_sha or recovery['inspect_function_sha256']!=function_sha:raise ValueError('Recovery inspect source changed')
 receipt_path=root/'resolved-receipts.jsonl'
 if digest(receipt_path)!=receipt_sha:raise ValueError('Recovery resolved receipt changed')
 receipts={}
 for record in journal(receipt_path):
  key=(record['split'],record['video'])
  if key in receipts:raise ValueError('Duplicate recovery receipt identity')
  receipts[key]=record
 parent=read(base/'parent-preflight.json')
 if decoded.get('targets_used_for_filtering') is not False or parent.get('recovery') is not None or parent.get('source_sha256')!=source_sha or parent.get('receipt_sha256')!=receipt_sha or parent.get('targets_used_for_filtering') is not False or decoded.get('source_sha256')!=source_sha or decoded.get('receipt_sha256')!=receipt_sha:raise ValueError('Recovery parent/source/receipt mismatch')
 old=parent['rows'];new=decoded['rows']
 if journal(base/'parent-rows.jsonl')!=old or len(old)!=len(new) or parent.get('total')!=len(old) or decoded.get('total')!=len(new) or parent.get('valid')!=sum(r.get('status')=='valid' for r in old):raise ValueError('Incomplete recovery parent coverage')
 if decoded.get('valid')!=sum(r.get('status')=='valid' for r in new) or len({r['sha256'] for r in old})!=len(old):raise ValueError('Duplicate recovery parent source')
 identity=lambda row:tuple(row[k] for k in ('split','video','sha256','num_segments'))
 if [identity(r) for r in old]!=[identity(r) for r in new]:raise ValueError('Recovery source order/hash/geometry changed')
 indices=[i for i,row in enumerate(old) if transient(row)];replacements=recovery['replacements']
 if not indices or [r.get('index') for r in replacements]!=indices:raise ValueError('Recovery replacement coverage differs')
 attempts=journal(base/'attempts.jsonl');cursor=0
 for i,row in enumerate(old):
  if i not in indices:
   if new[i]!=row:raise ValueError('Recovery changed unrelated row')
   continue
  replacement=replacements[indices.index(i)]
  if replacement!={'index':i,'parent_row_sha256':canonical(row),'replacement_row_sha256':canonical(new[i]),'raw_sha256':row['sha256'],'outcome':replacement.get('outcome')} or replacement.get('outcome') not in ('decoded_valid','confirmed_container_rejection'):raise ValueError('Recovery replacement hash mismatch')
  succeeded=False
  for number in range(1,4):
   if cursor>=len(attempts):raise ValueError('Missing recovery attempt')
   attempt=attempts[cursor];cursor+=1;result=attempt.get('result',{})
   descriptor=attempt.get('input_container',{});receipt=receipts.get((row['split'],row['video']),{})
   expected_container={'mode':'verified_mp4_copy','original_video':row['video'],'raw_sha256':row['sha256'],'source_bytes':receipt.get('bytes'),'copy_name':'source.mp4','copy_sha256':row['sha256'],'copy_bytes':receipt.get('bytes'),'retained_private':True}
   if receipt.get('status')!='complete' or receipt.get('sha256')!=row['sha256'] or type(receipt.get('bytes')) is not int or receipt['bytes']<=0 or not isinstance(descriptor,dict) or descriptor!=expected_container or type(descriptor.get('source_bytes')) is not int or type(descriptor.get('copy_bytes')) is not int or descriptor.get('retained_private') is not True:raise ValueError('Recovery input container/receipt mismatch')
   if set(attempt)!={'outcome','probe','input_container','parent_row_sha256','raw_sha256','attempt','result','parent_preflight_sha256','source_file_sha256','source_function_sha256'} or attempt['parent_row_sha256']!=canonical(row) or attempt['raw_sha256']!=row['sha256'] or type(attempt['attempt']) is not int or attempt['attempt']!=number or attempt['parent_preflight_sha256']!=recovery['parent_preflight_sha256'] or attempt['source_file_sha256']!=source_sha or attempt['source_function_sha256']!=function_sha or identity(result)!=identity(row):raise ValueError('Recovery attempt identity/order mismatch')
   outcome=attempt['outcome']
   if outcome=='decoded_valid':
    if replacement['outcome']!=outcome or attempt['probe'] is not None or result.get('status')!='valid' or result!=new[i] or not successful(result):raise ValueError('Recovery result differs from merged row')
    succeeded=True;break
   if outcome=='confirmed_container_rejection':
    if replacement['outcome']!=outcome or result!=new[i] or result.get('status')!='invalid' or result.get('last_frame_decoded') is not False or not result.get('error','').startswith('OSError:') or not structural_header_error(result['error']):raise ValueError('Recovery rejection lacks structural header evidence')
    checked_probe(attempt['probe'],row,receipt);succeeded=True;break
   if outcome is not None or attempt['probe'] is not None or not transient(result):raise ValueError('Recovery contains unresolved nontransient failure')
  if not succeeded:raise ValueError('Unresolved transient source')
 if cursor!=len(attempts):raise ValueError('Unexpected recovery audit attempts')
 return paths
