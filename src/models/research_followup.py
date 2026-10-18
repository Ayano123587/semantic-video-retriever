"""Freeze complete licensed-source research coverage before conditional training.

No test annotation file is read. Research source coverage is distinct from the
strict historical localization protocol and cannot relax its manifest pin.
"""
import argparse
import hashlib
import json
import math
import re
import copy
import numpy as np
import torch
from pathlib import Path, PurePosixPath
from src.data.didemo import HASHES, REVISION
from src.config import MODEL_ID, MODEL_REVISION
from src.models.encoder import WEIGHTS_SHA256
from src.data.research_receipts import reconcile
from src.models.moment_training import digest, load_cache, scores_for, CONFIGS as PHASE1_CONFIGS, SEEDS as PHASE1_SEEDS
from src.models.moment_training import train as phase1_train
from src.models.moments import evaluate, moments, interval
from src.models.moment_encoder import frame_schedule
from src.models.localization_followup import (LAMBDAS, ALPHAS, SEEDS,
                                             validate_protocol_grid, _train_checked, score_policy)

SELECTION_CANONICAL_SHA256 = 'ce430ba40063809c6bb885ee5bc7f8bb7da9db17ef4a52bb643c4215c1197c4a'
SELECTION_COUNTS = {'train': 6005, 'val': 825, 'test': 762}
SOURCES = ['models/resumable_training.py','config.py','data/didemo.py','models/moment_encoder.py','models/research_followup.py','models/localization_followup.py','models/moment_training.py',
           'models/positive_index.py','models/encoding_progress.py','models/moments.py','models/encoder.py',
           'data/decoder_preflight.py','data/decoder_recovery.py','data/prepare_moments.py','data/research_receipts.py']

def source_hashes():
    return {name:digest(Path(__file__).parents[1]/name) for name in SOURCES}


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf-8')).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def journal(path):
    content = Path(path).read_bytes()
    if content and not content.endswith(b'\n'):
        raise ValueError('Incomplete journal tail')
    return [json.loads(line) for line in content.splitlines() if line.strip()]


def accounting_path(name):
    """Resolve unchanged Windows/POSIX audit keys inside the staged data tree."""
    if not isinstance(name,str) or not name or ':' in name:
        raise ValueError('Unsafe source accounting path')
    relative=PurePosixPath(name.replace('\\','/'))
    if relative.is_absolute() or '..' in relative.parts or not relative.parts or relative.parts[0]!='data':
        raise ValueError('Unsafe source accounting path')
    base=Path.cwd().resolve();path=(base/Path(*relative.parts)).resolve()
    if not path.is_relative_to(base/'data'):
        raise ValueError('Source accounting path escapes data tree')
    return path


def checked_preparation(preparation, metadata='data/external/didemo'):
    """Recompute terminal coverage, exclusions, aliases and all train/val rows."""
    root, metadata = Path(preparation), Path(metadata)
    audit = read(root/'source-accounting.json')
    inputs = audit.get('input_sha256', {})
    if len(inputs) < 2:
        raise ValueError('Selection and complete source journal accounting required')
    source_paths = {name:accounting_path(name) for name in inputs}
    if len(set(source_paths.values())) != len(source_paths):raise ValueError('Duplicate normalized accounting path')
    for name, expected in inputs.items():
        if digest(source_paths[name]) != expected:
            raise ValueError('Changed source-accounting input')
    selections = []
    for name in inputs:
        try: value = read(source_paths[name])
        except json.JSONDecodeError: continue
        if canonical_sha(value) == SELECTION_CANONICAL_SHA256:selections.append(source_paths[name])
    if len(selections) != 1:
        raise ValueError('Full research selection fingerprint required')
    selection_path = selections[0]
    plan = read(selection_path)
    if plan.get('source', {}).get('revision') != REVISION or plan['source'].get('sha256') != HASHES:
        raise ValueError('Official research metadata identity mismatch')
    if {key:len(rows) for key,rows in plan['splits'].items()} != SELECTION_COUNTS:
        raise ValueError('Complete 7592-source selection required')
    if plan.get('use_terms', {}).get('noncommercial_research_only') is not True:
        raise ValueError('Research use terms required')
    others = [path for path in source_paths.values() if path != selection_path]
    rows = {str(path):journal(path) for path in others}
    recovery = []
    if audit.get('recovery_attempts',0):
        for path in others:
            candidate = rows[str(path)]
            original_rows = [row for other in others if other != path for row in rows[str(other)]]
            eligible = {(row['split'],row['video']) for row in original_rows if row['video'].endswith('.') and
                        row['status']=='failed' and row.get('reason')=='Invalid source video filename'}
            if candidate and len(candidate)==audit['recovery_attempts'] and {(r['split'],r['video']) for r in candidate}==eligible:
                recovery.append(path)
        if len(recovery)!=1:raise ValueError('Complete canonical ingestion recovery journal required')
    originals = [rows[str(path)] for path in others if path not in recovery]
    recovery_rows = rows[str(recovery[0])] if recovery else None
    if not recovery and any(row['video'].endswith('.') and row['status']=='failed' and row.get('reason')=='Invalid source video filename' for journal_rows in originals for row in journal_rows):
        raise ValueError('Known ingestion failures require explicit source-only recovery')
    resolved, recomputed = reconcile(plan, originals, recovery_rows)
    if recomputed['candidate_count'] != sum(SELECTION_COUNTS.values()):
        raise ValueError('Incomplete original source journals')
    if any(audit.get(key) != value for key,value in recomputed.items()):
        raise ValueError('Source accounting differs from terminal journals')
    receipt = root/'resolved-receipts.jsonl'
    if journal(receipt) != resolved:
        raise ValueError('Resolved receipts differ from terminal reconciliation')
    available = {(r['split'],r['video']):r for r in resolved if r['status']=='complete'}
    decoded_path = root/'decoder/preflight.json'
    decoded = read(decoded_path)
    if decoded.get('receipt_sha256') != digest(receipt) or decoded.get('source_sha256') != digest(Path(__file__).parents[1]/'data/decoder_preflight.py'):
        raise ValueError('Decoder receipt/source binding mismatch')
    if decoded.get('targets_used_for_filtering') is not False:
        raise ValueError('Decoder exclusions must not use targets')
    by_hash = {}
    for row in decoded['rows']:
        key = (row['split'],row['video'])
        if key not in available or available[key]['sha256'] != row['sha256'] or row['sha256'] in by_hash:
            raise ValueError('Decoder identity/hash coverage mismatch')
        if row.get('status') not in ('valid','invalid'):
            raise ValueError('Terminal decoder status required')
        if row['status']=='valid' and (row.get('last_frame_decoded') is not True or
            type(row.get('num_segments')) is not int or not 1 <= row['num_segments'] <= 6 or
            not all(isinstance(row.get(k),(int,float)) and math.isfinite(row[k]) and row[k]>0 for k in ('duration','fps')) or
            row['duration'] <= (row['num_segments']-1)*5):
            raise ValueError('Invalid successful decoder geometry')
        by_hash[row['sha256']] = row
    expected_hashes = {r['sha256'] for r in available.values()}
    if set(by_hash) != expected_hashes or decoded.get('total') != len(expected_hashes) or decoded.get('valid') != sum(r['status']=='valid' for r in by_hash.values()):
        raise ValueError('Incomplete decoder coverage')
    from src.data.decoder_recovery import checked_recovery
    checked_recovery(root,decoded,digest(receipt),Path(__file__).parents[1]/'data/decoder_preflight.py')
    prepared = root/'prepared'
    manifest = read(prepared/'manifest.json')
    if manifest.get('receipt_sha256') != digest(receipt) or manifest.get('selection_file_sha256') != digest(selection_path) or manifest.get('decoder_preflight_sha256') != digest(decoded_path):
        raise ValueError('Prepared source bindings mismatch')
    if manifest.get('source_revision') != REVISION or manifest.get('official_metadata_sha256') != HASHES:
        raise ValueError('Prepared official metadata mismatch')
    if set(manifest.get('files',{})) != {'media.json','train.json','val.json'} or (prepared/'test.json').exists():
        raise ValueError('Only train/validation annotations may be staged')
    for name, expected in manifest['files'].items():
        if digest(prepared/name) != expected:
            raise ValueError('Changed prepared file')
    invalid = sorted(key for key,row in by_hash.items() if row['status']=='invalid')
    if manifest.get('excluded_decoder_media_sha256') != invalid:
        raise ValueError('Decoder exclusions differ')
    retained = {key:r for key,r in available.items() if r['sha256'] not in invalid}
    groups = {}
    for key,row in retained.items():groups.setdefault(row['sha256'],[]).append(key)
    aliases = {key:min(group)[1] for group in groups.values() for key in group}
    media = read(prepared/'media.json')
    media_keys = {(r['split'],r['video']) for r in media}
    canonical = {(key[0],aliases[key]) for key in retained}
    if media_keys != canonical or len(media) != len(canonical):
        raise ValueError('Prepared media is an arbitrary or incomplete cohort')
    permitted = {(split,row['video']):row for split,rows in plan['splits'].items() for row in rows}
    for row in media:
        source = permitted[(row['split'],row['video'])]
        if any(row.get(key) != source.get(key) for key in ('license','aws_url','original_url')):
            raise ValueError('Prepared source rights/identity differs')
        original = retained[(row['split'],row['video'])]
        if row['sha256'] != original['sha256'] or row['bytes'] != original['bytes'] or row['num_segments'] != by_hash[row['sha256']]['num_segments']:
            raise ValueError('Prepared media identity/geometry differs')
    for split in ('train','val'):
        official_path = metadata/f'{split}_data.json'
        if digest(official_path) != HASHES[official_path.name]:
            raise ValueError('Changed official train/validation annotations')
        expected = []
        for i,row in enumerate(read(official_path)):
            key = (split,row['video'])
            if key not in retained:continue
            expected.append({'id':f'{split}:{i}:{row["annotation_id"]}', 'video':aliases[key],
                             'source_video':row['video'],'text':row['description'],
                             'times':[list(interval(t)) for t in row['times']]})
        if read(prepared/f'{split}.json') != expected:
            raise ValueError('Prepared queries omit or change official eligible annotations')
    return manifest, {'candidate_count':len(resolved),'retained_media':len(media),
                      'decoder_sources':len(expected_hashes),'decoder_excluded':len(invalid)}


def checked_research_cache(preparation, cache, metadata='data/external/didemo'):
    prepared_manifest, coverage = checked_preparation(preparation, metadata)
    prepared, cache = Path(preparation)/'prepared', Path(cache)
    manifest = read(cache/'manifest.json')
    encoder = manifest.get('encoder', {})
    if (encoder.get('model_id'),encoder.get('revision'),encoder.get('weights_sha256')) != (MODEL_ID,MODEL_REVISION,WEIGHTS_SHA256) or any(encoder.get('loading_info',{}).get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
        raise ValueError('Pinned complete CLIP encoder required')
    if manifest.get('status') != 'complete' or manifest.get('smoke_only') is not False or manifest.get('source_manifest_sha256') != digest(prepared/'manifest.json'):
        raise ValueError('Complete encoded full-preparation cache required')
    if manifest.get('test_text_or_targets_encoded') is not False or any(name.startswith('test') and name not in ('test-moments.npy','test-candidates.json') for name in manifest['files']):
        raise ValueError('Test annotations must not be encoded')
    for split in ('train','val'):
        if read(cache/f'{split}-queries.json') != read(prepared/f'{split}.json'):
            raise ValueError('Encoded queries differ from complete preparation')
    media = read(prepared/'media.json')
    ordered = [row for split in ('train','val','test') for row in media if row['split']==split]
    identity = manifest.get('encoding_identity')
    required = {'source_manifest_sha256','encoder','ordered_records_sha256','encoder_source_sha256',
                'progress_source_sha256','torch','device','model_dtype','transformers','numpy','ffmpeg','smoke_only',
                'pillow','imageio','imageio_ffmpeg','torch_build_sha256','cpu_identity'}
    if not isinstance(identity,dict) or set(identity) != required:
        raise ValueError('Complete encoding execution identity required')
    expected_identity = {'source_manifest_sha256':digest(prepared/'manifest.json'),'encoder':encoder,
                         'ordered_records_sha256':hashlib.sha256(json.dumps(ordered,sort_keys=True).encode()).hexdigest(),
                         'encoder_source_sha256':digest(Path(__file__).with_name('moment_encoder.py')),
                         'progress_source_sha256':digest(Path(__file__).with_name('encoding_progress.py')),
                         'model_dtype':'torch.float32','smoke_only':False}
    if any(identity.get(key) != value for key,value in expected_identity.items()):
        raise ValueError('Encoded execution/source/media identity mismatch')
    if not isinstance(identity['device'],str) or not re.fullmatch(r'cpu|cuda(?::[0-9]+)?',identity['device']) or any(not isinstance(identity[key],str) or not identity[key].strip() for key in ('torch','transformers','numpy','ffmpeg','pillow','imageio','imageio_ffmpeg')):
        raise ValueError('Recorded encoder device/environment identity required')
    cpu=identity['cpu_identity']
    if not isinstance(identity['torch_build_sha256'],str) or not re.fullmatch('[0-9a-f]{64}',identity['torch_build_sha256']):
        raise ValueError('Recorded encoder Torch build identity required')
    if not isinstance(cpu,dict) or not {'machine','processor'}.issubset(cpu) or set(cpu)-{'machine','processor','cpuinfo'} or not isinstance(cpu['machine'],str) or not cpu['machine'].strip() or not isinstance(cpu['processor'],str):
        raise ValueError('Recorded encoder CPU identity required')
    if 'cpuinfo' in cpu and (not isinstance(cpu['cpuinfo'],dict) or not cpu['cpuinfo'] or any(not isinstance(key,str) or not isinstance(value,str) or not value.strip() for key,value in cpu['cpuinfo'].items())):
        raise ValueError('Recorded encoder CPU instruction identity required')
    frames = read(cache/'frames.json')
    if len(frames) != len(media) or {(r['split'],r['video'],r['source_sha256']) for r in frames} != {(r['split'],r['video'],r['sha256']) for r in media}:
        raise ValueError('Encoded media coverage differs from complete preparation')
    decoded = {r['sha256']:r for r in read(Path(preparation)/'decoder/preflight.json')['rows']}
    media_lookup = {(r['split'],r['video']):r for r in media}
    for row in frames:
        source = media_lookup[(row['split'],row['video'])]; verified = decoded[source['sha256']]
        if any(row.get(key) != verified.get(key) for key in ('duration','fps')) or row.get('samples') != frame_schedule(source['num_segments'],verified['duration'],verified['fps']):
            raise ValueError('Encoded frame timing differs from verified source')
    expected_files = {'frames.json',*[f'{split}-{kind}' for split in ('train','val','test') for kind in ('moments.npy','candidates.json')],*[f'{split}-{kind}' for split in ('train','val') for kind in ('text.npy','queries.json')]}
    if set(manifest['files']) != expected_files:
        raise ValueError('Complete media features and train/validation text only required')
    for split in ('train','val','test'):
        expected = [{'video':r['video'],'moment':list(moment)} for r in media if r['split']==split for moment in moments(r['num_segments'])]
        if read(cache/f'{split}-candidates.json') != expected:
            raise ValueError('Encoded candidates omit complete moment geometry')
        values=np.load(cache/f'{split}-moments.npy',allow_pickle=False,mmap_mode='r')
        if values.shape != (len(expected),512) or values.dtype != np.float32:
            raise ValueError('Complete encoded moment feature geometry required')
        for offset in range(0,len(values),1024):
            part=values[offset:offset+1024]
            if not np.isfinite(part).all() or not np.allclose(np.linalg.norm(part,axis=1),1,atol=1e-4):
                raise ValueError('Finite unit complete moment features required')
    data, loaded = load_cache(cache,indexed=True)
    if loaded != manifest:raise ValueError('Cache changed during validation')
    return data, manifest, coverage


def freeze(preparation, cache, output, metadata='data/external/didemo'):
    output = Path(output)
    if output.exists():raise ValueError('Fresh frozen protocol required')
    data, manifest, coverage = checked_research_cache(preparation,cache,metadata)
    val = data['val']
    baseline = evaluate(scores_for([],val['x'],val['v']),val['candidates'],val['queries'])
    protocol = {'schema':'research_conditional_v1','implementation_sha256':source_hashes(),'source_manifest_sha256':digest(Path(cache)/'manifest.json'),
                'preparation_manifest_sha256':digest(Path(preparation)/'prepared/manifest.json'),
                'accounting_sha256':digest(Path(preparation)/'source-accounting.json'),
                'resolved_receipts_sha256':digest(Path(preparation)/'resolved-receipts.jsonl'),
                'decoder_sha256':digest(Path(preparation)/'decoder/preflight.json'),
                'coverage':coverage,'local_loss_weights':LAMBDAS,'adapted_score_weights':ALPHAS,'seeds':SEEDS,
                'optimizer':{'name':'AdamW','lr':.0003,'weight_decay':.01,'temperature':.07,'batch':64},
                'epochs':{'maximum':50,'minimum':5,'patience':8,'selection':'per-seed validation primary; earliest exact ties'},
                'promotion':{'minimum_absolute_primary_gain_over_frozen_clip':.01,'maximum_within_video_miou_drop':.01},
                'phase1':{'configs':PHASE1_CONFIGS,'seeds':PHASE1_SEEDS,'temperature':.07,'batch':64,
                          'epochs':{'maximum':50,'minimum':5,'patience':8},
                          'promotion':{'minimum_absolute_primary_gain_over_frozen_clip':.01,'maximum_within_video_miou_drop':.01}},
                'baseline':baseline,'encoding_identity':manifest['encoding_identity'],'test_annotations_used':False,
                'global_selection':{'phases':['phase1','phase2'],'criterion':'highest eligible validation corpus_consensus_r1_iou50',
                                    'ties':'frozen_clip then phase1 then phase2; fixed within-phase policy order',
                                    'test_access':False}}
    validate_protocol_grid(protocol)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8',newline='\n') as stream:json.dump(protocol,stream,indent=2,allow_nan=False)
    return protocol


def checked_protocol(preparation, cache, protocol, protocol_sha256, metadata='data/external/didemo'):
    if digest(protocol) != protocol_sha256:raise ValueError('Frozen protocol fingerprint mismatch')
    p = read(protocol)
    if p.get('schema') != 'research_conditional_v1' or p.get('test_annotations_used') is not False:
        raise ValueError('Research-only frozen protocol required')
    validate_protocol_grid(p)
    expected_phase1={'configs':PHASE1_CONFIGS,'seeds':PHASE1_SEEDS,'temperature':.07,'batch':64,
                     'epochs':{'maximum':50,'minimum':5,'patience':8},
                     'promotion':{'minimum_absolute_primary_gain_over_frozen_clip':.01,'maximum_within_video_miou_drop':.01}}
    if p.get('phase1') != expected_phase1:raise ValueError('Frozen phase1 grid mismatch')
    if p.get('implementation_sha256') != source_hashes():raise ValueError('Frozen implementation changed')
    bindings = {'source_manifest_sha256':Path(cache)/'manifest.json',
                'preparation_manifest_sha256':Path(preparation)/'prepared/manifest.json',
                'accounting_sha256':Path(preparation)/'source-accounting.json',
                'resolved_receipts_sha256':Path(preparation)/'resolved-receipts.jsonl',
                'decoder_sha256':Path(preparation)/'decoder/preflight.json'}
    if any(p.get(key) != digest(path) for key,path in bindings.items()):
        raise ValueError('Frozen cohort/cache bindings changed')
    data, manifest, coverage = checked_research_cache(preparation,cache,metadata)
    if p.get('coverage') != coverage:raise ValueError('Frozen source coverage changed')
    if p.get('encoding_identity') != manifest.get('encoding_identity'):raise ValueError('Frozen encoding execution identity changed')
    return data,manifest,p


def train(preparation, cache, output, protocol, protocol_sha256, metadata='data/external/didemo'):
    if Path(output).exists():raise ValueError('Fresh training output required')
    data,manifest,p=checked_protocol(preparation,cache,protocol,protocol_sha256,metadata)
    return _train_checked(data,manifest,p,output,protocol,p['source_manifest_sha256'])


def train_phase1(preparation, cache, output, protocol, protocol_sha256, metadata='data/external/didemo'):
    if Path(output).exists():raise ValueError('Fresh phase1 output required')
    data,manifest,p=checked_protocol(preparation,cache,protocol,protocol_sha256,metadata)
    val=data['val']
    baseline=evaluate(scores_for([],val['x'],val['v']),val['candidates'],val['queries'])
    if not metrics_match(baseline,p['baseline']):raise ValueError('Frozen phase1 baseline mismatch')
    del val,data,manifest
    result=phase1_train(cache,output)
    if not metrics_match(result['baseline'],baseline) or result.get('cache_manifest_sha256') != p['source_manifest_sha256']:
        raise ValueError('Phase1 training cohort changed')
    result['protocol_sha256']=protocol_sha256
    result['full_coverage']=p['coverage']
    report=Path(output)/'results.json'
    # Existing trainer owns this newly-created report; attach the checked binding
    # only after successful terminal training, without changing any scores.
    report.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    return result


def metrics_match(actual, expected):
    if set(actual) != set(expected):return False
    for key,value in expected.items():
        other = actual[key]
        if isinstance(value,(int,float)) and not isinstance(value,bool):
            if not isinstance(other,(int,float)) or not math.isfinite(other) or abs(other-value)>1e-10:return False
        elif other != value:return False
    return True


def choose(preparation, cache, phase1, phase2, output, protocol, protocol_sha256,
           phase1_report_sha256, phase2_report_sha256, metadata='data/external/didemo'):
    """Replay both saved policies on the same validation cohort, then freeze one."""
    output = Path(output)
    if output.exists():raise ValueError('Fresh global selection output required')
    if digest(protocol) != protocol_sha256:raise ValueError('Frozen protocol fingerprint mismatch')
    p = read(protocol)
    if p.get('global_selection') != {'phases':['phase1','phase2'],'criterion':'highest eligible validation corpus_consensus_r1_iou50',
                                   'ties':'frozen_clip then phase1 then phase2; fixed within-phase policy order','test_access':False}:
        raise ValueError('Prespecified global choice required')
    data,manifest,p=checked_protocol(preparation,cache,protocol,protocol_sha256,metadata)
    val = data['val']
    baseline = evaluate(scores_for([],val['x'],val['v']),val['candidates'],val['queries'])
    if not metrics_match(baseline,p['baseline']):raise ValueError('Frozen baseline mismatch')
    winner = {'schema':1,'states':[],'adapted_score_weight':0.,'encoder':manifest['encoder'],
              'cache_manifest_sha256':p['source_manifest_sha256'],'winner':'frozen_clip','validation':baseline,'test_targets_used':False}
    best = baseline['corpus_consensus_r1_iou50']; evidence = []
    for name,directory,expected_sha in [('phase1',phase1,phase1_report_sha256),('phase2',phase2,phase2_report_sha256)]:
        directory = Path(directory);report_path = directory/'results.json';checkpoint = directory/'selected.pt'
        if digest(report_path) != expected_sha:raise ValueError('Changed phase report')
        report = read(report_path)
        if report.get('selected_checkpoint_sha256') != digest(checkpoint) or report.get('cache_manifest_sha256') != p['source_manifest_sha256']:
            raise ValueError('Phase checkpoint/cache binding mismatch')
        if report.get('test_targets_used') is not False or not metrics_match(report['baseline'],baseline):
            raise ValueError('Phase baseline or no-test contract mismatch')
        if report.get('protocol_sha256') != protocol_sha256:
            raise ValueError('Phase used a different frozen protocol')
        bundle = torch.load(checkpoint,map_location='cpu',weights_only=True)
        if bundle.get('cache_manifest_sha256') != p['source_manifest_sha256'] or bundle.get('encoder') != manifest['encoder'] or bundle.get('test_targets_used') is not False:
            raise ValueError('Phase checkpoint identity mismatch')
        measured = evaluate(score_policy(bundle,val['x'],val['v']),val['candidates'],val['queries'])
        if not metrics_match(measured,report['selected_validation']) or not metrics_match(measured,bundle['validation']):
            raise ValueError('Phase selected metrics do not replay')
        primary = measured['corpus_consensus_r1_iou50']
        eligible = primary-baseline['corpus_consensus_r1_iou50']>=.01-1e-12 and measured['within_video_official_miou']>=baseline['within_video_official_miou']-.01-1e-12
        evidence.append({'phase':name,'report_sha256':expected_sha,'checkpoint_sha256':digest(checkpoint),
                         'validation':measured,'promotion_eligible':eligible})
        if eligible and primary>best:
            winner = copy.deepcopy(bundle);winner['winner']=name+':'+report['selected'];best=primary
    winner['global_protocol_sha256']=protocol_sha256
    output.mkdir(parents=True)
    with (output/'selected.pt').open('xb') as stream:torch.save(winner,stream)
    restored=torch.load(output/'selected.pt',map_location='cpu',weights_only=True)
    delta=float(np.max(np.abs(score_policy(winner,val['x'][:3],val['v'])-score_policy(restored,val['x'][:3],val['v']))))
    if delta>1e-6:raise ValueError('Global saved-policy replay mismatch')
    result={'selected':winner['winner'],'selected_validation':winner['validation'],'baseline':baseline,
            'phases':evidence,'protocol_sha256':protocol_sha256,'cache_manifest_sha256':p['source_manifest_sha256'],
            'selected_checkpoint_sha256':digest(output/'selected.pt'),'checkpoint_replay_max_abs_error':delta,'test_targets_used':False}
    with (output/'results.json').open('x',encoding='utf-8',newline='\n') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['freeze','train','train-phase1','choose'])
    parser.add_argument('--preparation',required=True);parser.add_argument('--cache',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--metadata',default='data/external/didemo')
    parser.add_argument('--protocol');parser.add_argument('--protocol-sha256')
    parser.add_argument('--phase1');parser.add_argument('--phase2')
    parser.add_argument('--phase1-report-sha256');parser.add_argument('--phase2-report-sha256')
    args = parser.parse_args()
    if args.action=='freeze':result=freeze(args.preparation,args.cache,args.output,args.metadata)
    else:
        if not args.protocol or not args.protocol_sha256:parser.error('train requires frozen --protocol and --protocol-sha256')
        if args.action=='choose':
            if not all([args.phase1,args.phase2,args.phase1_report_sha256,args.phase2_report_sha256]):parser.error('choose requires both phase directories and report hashes')
            result=choose(args.preparation,args.cache,args.phase1,args.phase2,args.output,args.protocol,args.protocol_sha256,args.phase1_report_sha256,args.phase2_report_sha256,args.metadata)
        else:
            runner=train_phase1 if args.action=='train-phase1' else train
            result=runner(args.preparation,args.cache,args.output,args.protocol,args.protocol_sha256,args.metadata)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
