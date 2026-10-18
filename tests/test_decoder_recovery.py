import copy,json
from pathlib import Path
import pytest
from src.data import decoder_recovery as r

def put(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8');return path

def lines(path,rows):path.write_text(''.join(json.dumps(row)+'\n' for row in rows),encoding='utf-8');return path

def fixture(tmp_path):
 source=tmp_path/'decoder.py';source.write_text('def inspect(record,path):\n    return record\n',encoding='utf-8')
 base=tmp_path/'decoder/recovery';base.mkdir(parents=True)
 old=[{'split':'test','video':'v.mp4','sha256':'a'*64,'num_segments':1,'status':'invalid','error':'PermissionError: [WinError 32] locked','last_frame_decoded':False}, {'split':'train','video':'w.mp4','sha256':'b'*64,'num_segments':1,'status':'valid','duration':5.,'fps':25.,'last_frame_decoded':True,'last_frame':93}]
 receipt_path=lines(tmp_path/'resolved-receipts.jsonl',[{'split':row['split'],'video':row['video'],'status':'complete','sha256':row['sha256'],'bytes':11} for row in old]);receipt_sha=r.digest(receipt_path)
 parent={'rows':old,'total':2,'valid':1,'source_sha256':r.digest(source),'receipt_sha256':receipt_sha,'targets_used_for_filtering':False}
 put(base/'parent-preflight.json',parent);lines(base/'parent-rows.jsonl',old);(base/'recover.py').write_text('# Plain private helper snapshot\n',encoding='utf-8')
 new=copy.deepcopy(old);new[0]={'split':'test','video':'v.mp4','sha256':'a'*64,'num_segments':1,'status':'valid','duration':5.,'fps':25.,'last_frame_decoded':True,'last_frame':93}
 container={'mode':'verified_mp4_copy','original_video':old[0]['video'],'raw_sha256':old[0]['sha256'],'source_bytes':11,'copy_name':'source.mp4','copy_sha256':old[0]['sha256'],'copy_bytes':11,'retained_private':True}
 attempt={'outcome':'decoded_valid','probe':None,'input_container':container,'parent_row_sha256':r.canonical(old[0]),'raw_sha256':'a'*64,'attempt':1,'result':new[0],'parent_preflight_sha256':r.digest(base/'parent-preflight.json'),'source_file_sha256':r.digest(source),'source_function_sha256':r.function_hash(source)}
 lines(base/'attempts.jsonl',[attempt])
 recovery={field:r.digest(base/name) for field,name in r.FILES.items()};recovery.update(unchanged_inspect_source_sha256=r.digest(source),inspect_function_sha256=r.function_hash(source),targets_used_for_filtering=False,replacements=[{'index':0,'parent_row_sha256':r.canonical(old[0]),'replacement_row_sha256':r.canonical(new[0]),'raw_sha256':'a'*64,'outcome':'decoded_valid'}])
 merged={**parent,'rows':new,'valid':2,'recovery':recovery}
 return source,base,merged

def test_optional_proof_and_exact_artifacts(tmp_path):
 assert r.checked_recovery(tmp_path,{},'receipt',tmp_path/'absent.py')==[]
 source,base,merged=fixture(tmp_path)
 assert r.checked_recovery(tmp_path,merged,merged['receipt_sha256'],source)==[base/name for name in r.FILES.values()]

@pytest.mark.parametrize('defect',['helper','parent_hash','parent_receipt','parent_journal','source','nontransient','duplicate','unrelated','replacement','attempt','result','extra','missing','geometry'])
def test_tampered_recovery_rejected_even_rehashed(tmp_path,defect):
 source,base,merged=fixture(tmp_path);rec=merged['recovery']
 if defect=='helper':(base/'recover.py').write_text('changed')
 elif defect=='parent_hash':rec['parent_preflight_sha256']='wrong'
 elif defect=='source':source.write_text('def inspect(record,path):\n return None\n')
 elif defect=='parent_journal':lines(base/'parent-rows.jsonl',[]);rec['parent_rows_sha256']=r.digest(base/'parent-rows.jsonl')
 elif defect in ('parent_receipt','nontransient','duplicate'):
  parent=r.read(base/'parent-preflight.json')
  if defect=='parent_receipt':parent['receipt_sha256']='different'
  elif defect=='nontransient':parent['rows'][0]['error']='ValueError: corrupt'
  else:parent['rows'][1]['sha256']='a'*64
  put(base/'parent-preflight.json',parent);lines(base/'parent-rows.jsonl',parent['rows']);rec['parent_preflight_sha256']=r.digest(base/'parent-preflight.json');rec['parent_rows_sha256']=r.digest(base/'parent-rows.jsonl')
 elif defect=='unrelated':merged['rows'][1]['duration']=6.
 elif defect=='geometry':merged['rows'][0]['num_segments']=2
 elif defect=='replacement':rec['replacements'][0]['index']=1
 else:
  attempts=r.journal(base/'attempts.jsonl')
  if defect=='attempt':attempts[0]['attempt']=2
  elif defect=='result':attempts[0]['result']['duration']=6.
  elif defect=='extra':attempts.append(copy.deepcopy(attempts[0]))
  else:attempts=[]
  lines(base/'attempts.jsonl',attempts);rec['audit_sha256']=r.digest(base/'attempts.jsonl')
 with pytest.raises(ValueError):r.checked_recovery(tmp_path,merged,merged['receipt_sha256'],source)


@pytest.mark.parametrize('field,value',[('mode','original_path'),('original_video','other.mp4'),('raw_sha256','c'*64),('source_bytes',12),('source_bytes',True),('copy_name','../source.mp4'),('copy_sha256','c'*64),('copy_bytes',12),('copy_bytes',True),('retained_private',False),('extra','untrusted')])
def test_copy_descriptor_tamper_rejected_after_audit_rehash(tmp_path,field,value):
 source,base,merged=fixture(tmp_path)
 attempts=r.journal(base/'attempts.jsonl');attempts[0]['input_container'][field]=value
 lines(base/'attempts.jsonl',attempts);merged['recovery']['audit_sha256']=r.digest(base/'attempts.jsonl')
 with pytest.raises(ValueError,match='container'):r.checked_recovery(tmp_path,merged,merged['receipt_sha256'],source)

def test_old_attempt_schema_is_not_silently_reinterpreted(tmp_path):
 source,base,merged=fixture(tmp_path);attempts=r.journal(base/'attempts.jsonl');attempts[0].pop('input_container')
 lines(base/'attempts.jsonl',attempts);merged['recovery']['audit_sha256']=r.digest(base/'attempts.jsonl')
 with pytest.raises(ValueError):r.checked_recovery(tmp_path,merged,merged['receipt_sha256'],source)

def rejection_fixture(tmp_path):
 source,base,merged=fixture(tmp_path)
 diagnostic='stream 1, contradictionary STSC and STCO\nerror reading header\nInvalid data found when processing input'
 rejected={**merged['rows'][0],'status':'invalid','duration':0.,'fps':0.,'last_frame_decoded':False,'error':'OSError: '+diagnostic};rejected.pop('last_frame')
 merged['rows'][0]=rejected;merged['valid']=1
 p={'command':['/ffmpeg','-v','error','-i','/original/v.mp4','-map','0:v:0','-frames:v','1','-f','null','-'],'raw_sha256':'a'*64,'source_bytes':11,'ffmpeg_executable':'/ffmpeg','ffmpeg_executable_sha256':r.HEADER_PROBE_EXECUTABLE_SHA256,'ffmpeg_version_stdout':'ffmpeg version fixture','ffmpeg_version_stderr':'','exit_code':1,'stdout':'','stderr':diagnostic,'source_unchanged_after':True,'media_rewritten':False,'probe':'first_video_frame_to_null'}
 text=json.dumps(p,indent=2);probe={'receipt_json':text,'receipt_sha256':__import__('hashlib').sha256(text.encode()).hexdigest()}
 attempts=r.journal(base/'attempts.jsonl');attempts[0].update(outcome='confirmed_container_rejection',probe=probe,result=rejected);lines(base/'attempts.jsonl',attempts)
 rec=merged['recovery'];rec['audit_sha256']=r.digest(base/'attempts.jsonl');rec['replacements'][0].update(outcome='confirmed_container_rejection',replacement_row_sha256=r.canonical(rejected))
 return source,base,merged

def test_confirmed_container_rejection_keeps_invalid_count(tmp_path):
 source,base,merged=rejection_fixture(tmp_path)
 assert r.checked_recovery(tmp_path,merged,merged['receipt_sha256'],source)
 assert merged['valid']==1 and merged['rows'][0]['status']=='invalid'

@pytest.mark.parametrize('defect',['probe_hash','raw_sha256','source_bytes','source_bool','command','exit_code','rewritten','diagnostic','executable_sha','missing_probe','generic_inspect','resource_inspect','outcome'])
def test_rejection_cannot_hide_generic_or_forged_evidence(tmp_path,defect):
 source,base,merged=rejection_fixture(tmp_path);attempts=r.journal(base/'attempts.jsonl');a=attempts[0]
 if defect=='probe_hash':a['probe']['receipt_sha256']='0'*64
 elif defect=='missing_probe':a['probe']=None
 elif defect in ('generic_inspect','resource_inspect'):
  text='OSError: missing codec' if defect=='generic_inspect' else a['result']['error']+' Resource temporarily unavailable'
  a['result']['error']=text;merged['rows'][0]['error']=text;merged['recovery']['replacements'][0]['replacement_row_sha256']=r.canonical(merged['rows'][0])
 elif defect=='outcome':a['outcome']='decoded_valid'
 else:
  p=json.loads(a['probe']['receipt_json'])
  if defect=='raw_sha256':p['raw_sha256']='b'*64
  elif defect=='source_bytes':p['source_bytes']=12
  elif defect=='source_bool':p['source_bytes']=True
  elif defect=='command':p['command'][4]='/original/other.mp4'
  elif defect=='exit_code':p['exit_code']=0
  elif defect=='rewritten':p['media_rewritten']=True
  elif defect=='diagnostic':p['stderr']='Permission denied'
  else:p['ffmpeg_executable_sha256']='c'*64
  text=json.dumps(p);a['probe']={'receipt_json':text,'receipt_sha256':__import__('hashlib').sha256(text.encode()).hexdigest()}
 lines(base/'attempts.jsonl',attempts);merged['recovery']['audit_sha256']=r.digest(base/'attempts.jsonl')
 with pytest.raises(ValueError):r.checked_recovery(tmp_path,merged,merged['receipt_sha256'],source)
