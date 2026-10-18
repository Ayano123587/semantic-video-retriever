"""Encode authentic DiDeMo videos and train/validation text; no test targets needed."""
import argparse,hashlib,json,math,time
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from src.models.encoder import CLIPEncoder
from src.models.moments import moments
from src.data.didemo import download_one,media_filename,decoder_source


def digest(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def cpu_identity():
    """Stable instruction-set identity; omit changing clocks and processor IDs."""
    import platform
    result={'machine':platform.machine(),'processor':platform.processor()}
    path=Path('/proc/cpuinfo')
    if path.is_file():
        first=path.read_text(encoding='utf-8').split('\n\n',1)[0]
        wanted={'vendor_id','model name','cpu family','model','stepping','flags'}
        result['cpuinfo']={key.strip():value.strip() for line in first.splitlines()
                           if ':' in line for key,value in [line.split(':',1)] if key.strip() in wanted}
    return result


def frame_schedule(segments,duration,fps):
    moments(segments)
    if not all(math.isfinite(v) and v>0 for v in (duration,fps)) or duration <= (segments-1)*5:
        raise ValueError('Invalid or annotation-inconsistent duration/FPS')
    schedule=[]
    for segment in range(segments):
        start=segment*5.;end=min((segment+1)*5.,duration)
        for fraction in (.25,.75):
            timestamp=start+(end-start)*fraction
            frame=int(math.floor(timestamp*fps))
            schedule.append({'segment':segment,'frame':frame,'seconds':frame/fps})
    return schedule


@contextmanager
def verified_media(record, media, stream_download=False):
    """Keep only one verified source on disk when using cloud research media."""
    media=Path(media)
    if not stream_download:
        path=media/media_filename(record['video'])
        if digest(path)!=record['sha256']:raise ValueError('Changed media source')
        yield path
        return
    media.mkdir(parents=True,exist_ok=True)
    # Cleanup is restricted to this newly created owned directory; existing
    # user media and other concurrent downloads are never removed.
    with TemporaryDirectory(prefix='verified-source-',dir=media) as temporary:
        receipt=download_one(record,temporary)
        if receipt['sha256']!=record['sha256']:raise ValueError('Cloud media differs from frozen source')
        yield Path(temporary)/media_filename(record['video'])


def encode(source,media,output,smoke=False,device='cpu',*,stream_download=False,resume_from=None,max_seconds=None):
    import imageio.v2 as imageio
    from PIL import Image
    source=Path(source);media=Path(media);output=Path(output)
    if max_seconds is not None and (isinstance(max_seconds,bool) or not math.isfinite(max_seconds) or max_seconds<=0):
        raise ValueError('Positive finite service-window duration required')
    if output.exists():raise ValueError('Refuse to overwrite encoder cache')
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    for name,expected in manifest['files'].items():
        if digest(source/name)!=expected:raise ValueError('Changed staged source: '+name)
    records=json.loads((source/'media.json').read_text(encoding='utf-8'))
    queries={s:json.loads((source/f'{s}.json').read_text(encoding='utf-8')) for s in ('train','val')}
    if smoke:
        chosen={s:next(r['video'] for r in records if r['split']==s) for s in ('train','val')}
        records=[r for r in records if r['split'] in chosen and r['video']==chosen[r['split']]]
        queries={s:[q for q in rows if q['video']==chosen[s]][:2] for s,rows in queries.items()}
    if any(r.get('split') not in ('train','val','test') for r in records):
        raise ValueError('Unknown prepared media split')
    records=[r for split in ('train','val','test') for r in records if r['split']==split]
    output.mkdir(parents=True,exist_ok=False)
    encoder=CLIPEncoder(device);start=time.monotonic();vectors={};candidates={};frame_receipts=[]
    from importlib.metadata import version
    import imageio_ffmpeg
    from src.models.encoding_progress import EncodingProgress
    identities={'source_manifest_sha256':digest(source/'manifest.json'),'encoder':encoder.receipt,
        'ordered_records_sha256':hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest(),
        'encoder_source_sha256':digest(__file__),
        'progress_source_sha256':digest(Path(__file__).with_name('encoding_progress.py')),
        'torch':str(encoder.torch.__version__),'device':str(device),'model_dtype':str(encoder.model.dtype),
        'transformers':version('transformers'),
        'pillow':version('pillow'),'imageio':version('imageio'),'imageio_ffmpeg':version('imageio-ffmpeg'),
        'torch_build_sha256':hashlib.sha256(encoder.torch.__config__.show().encode('utf-8')).hexdigest(),
        'cpu_identity':cpu_identity(),
        'numpy':np.__version__,'ffmpeg':imageio_ffmpeg.get_ffmpeg_version(),'smoke_only':smoke}
    progress=EncodingProgress(output/'encoding-progress',identities,records,resume_from)
    resumed_videos=len(progress.entries)
    for split in ('train','val','test'):
        embeddings=[];rows=[]
        for record in records:
            if record['split']!=split:continue
            index=len(frame_receipts)
            reused=progress.get(index)
            if reused is None:
                if max_seconds is not None and time.monotonic()-start>=max_seconds and progress.entries:
                    partial={'status':'partial','encoded_videos':len(progress.entries),'total_videos':len(records),
                        'source_manifest_sha256':identities['source_manifest_sha256'],
                        'reason':'Service-window checkpoint; remaining sources must be resumed, not discarded.',
                        'test_text_or_targets_encoded':False}
                    (output/'progress-manifest.json').write_text(json.dumps(partial,indent=2)+'\n',encoding='utf-8')
                    print(json.dumps(partial),flush=True);return partial
                with verified_media(record,media,stream_download) as original,decoder_source(original) as path:
                    reader=imageio.get_reader(path,format='ffmpeg')
                    try:
                        meta=reader.get_meta_data();fps=float(meta['fps']);duration=float(meta['duration'])
                        schedule=frame_schedule(record['num_segments'],duration,fps)
                        images=[Image.fromarray(reader.get_data(item['frame'])).convert('RGB') for item in schedule]
                    finally:reader.close()
                    frame_values=encoder.images(images)
                frame_receipt={'video':record['video'],'split':split,'source_sha256':record['sha256'],
                    'duration':duration,'fps':fps,'samples':schedule}
                progress.append(index,frame_values,frame_receipt)
            else:
                frame_values,frame_receipt=reused
            segments=frame_values.reshape(record['num_segments'],2,-1).mean(axis=1)
            for a,b in moments(record['num_segments']):
                value=segments[a:b+1].mean(axis=0);value=value/np.linalg.norm(value)
                embeddings.append(value);rows.append({'video':record['video'],'moment':[a,b]})
            frame_receipts.append(frame_receipt)
            print(json.dumps({'encoded_videos':len(frame_receipts),'total_videos':len(records)}),flush=True)
        if embeddings:
            vectors[split]=np.stack(embeddings).astype(np.float32);candidates[split]=rows
    file_hashes={};truncation={}
    for split in vectors:
        path=output/f'{split}-moments.npy';np.save(path,vectors[split],allow_pickle=False);file_hashes[path.name]=digest(path)
        path=output/f'{split}-candidates.json';path.write_bytes((json.dumps(candidates[split],indent=2)+'\n').encode());file_hashes[path.name]=digest(path)
    for split,rows in queries.items():
        texts=[q['text'] for q in rows];pieces=[]
        for offset in range(0,len(texts),32):pieces.append(encoder.texts(texts[offset:offset+32]))
        values=np.concatenate(pieces)
        lengths=[len(ids) for ids in encoder.processor.tokenizer(texts,truncation=False)['input_ids']]
        truncation[split]={'queries':len(rows),'over_token_limit':sum(n>77 for n in lengths),'maximum_tokens':max(lengths)}
        path=output/f'{split}-text.npy';np.save(path,values,allow_pickle=False);file_hashes[path.name]=digest(path)
        path=output/f'{split}-queries.json';path.write_bytes((json.dumps(rows,ensure_ascii=False,indent=2)+'\n').encode('utf-8'));file_hashes[path.name]=digest(path)
    # Actual pretrained output replay after disk serialization.
    replay_query=queries['val'][0]['text'];replay=encoder.text(replay_query)
    saved=np.load(output/'val-text.npy',allow_pickle=False)[0]
    error=float(np.max(np.abs(replay-saved)))
    if error>1e-5:raise ValueError('Saved pretrained embedding replay mismatch')
    path=output/'frames.json';path.write_bytes((json.dumps(frame_receipts,indent=2)+'\n').encode());file_hashes[path.name]=digest(path)
    receipt={'status':'complete','encoder':encoder.receipt,'files':file_hashes,'source_manifest_sha256':digest(source/'manifest.json'),
        'encoding_identity':identities,
        'smoke_only':smoke,'seconds':time.monotonic()-start,'videos':len(records),'text_truncation':truncation,
        'actual_text_replay_max_abs_error':error,'test_text_or_targets_encoded':False,
        'media_transport':'verified temporary source per video' if stream_download else 'verified existing local media',
        'resumed_videos':resumed_videos,
        'splits':{s:{'moments':len(v),'dimension':v.shape[1]} for s,v in vectors.items()}}
    (output/'manifest.json').write_bytes((json.dumps(receipt,indent=2,allow_nan=False)+'\n').encode())
    print(json.dumps(receipt,indent=2));return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--media',required=True)
    p.add_argument('--output',required=True);p.add_argument('--smoke',action='store_true')
    p.add_argument('--stream-download',action='store_true');p.add_argument('--resume-from')
    p.add_argument('--max-seconds',type=float);a=p.parse_args()
    encode(a.source,a.media,a.output,a.smoke,stream_download=a.stream_download,resume_from=a.resume_from,max_seconds=a.max_seconds)
