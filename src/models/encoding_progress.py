"""Hash-bound numeric encoder checkpoints across cloud session windows."""
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import numpy as np


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


class EncodingProgress:
    """Only complete ordered entries are reusable; no executable serialization."""
    def __init__(self, directory, identities, records, resume_from=None):
        self.directory=Path(directory)
        if self.directory.exists():raise ValueError('Fresh progress destination required')
        self.directory.mkdir(parents=True)
        self.records=records;self.entries=[];self.interrupted_tail_bytes=0
        expected={'schema':1,'identities':identities}
        if resume_from is not None:
            previous=Path(resume_from)
            if json.loads((previous/'identity.json').read_text(encoding='utf-8'))!=expected:
                raise ValueError('Encoder resume identity changed')
            raw=(previous/'complete.jsonl').read_bytes()
            boundary=raw.rfind(b'\n')+1
            complete,tail=raw[:boundary],raw[boundary:]
            self.interrupted_tail_bytes=len(tail)
            if tail:(self.directory/'interrupted-tail.part').write_bytes(tail)
            for line in complete.splitlines():
                entry=json.loads(line)
                self._validate_entry(entry,previous)
                shutil.copyfile(previous/entry['file'],self.directory/entry['file'])
                self.entries.append(entry)
        (self.directory/'identity.json').write_text(json.dumps(expected,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        (self.directory/'complete.jsonl').write_text(''.join(json.dumps(e,allow_nan=False)+'\n' for e in self.entries),encoding='utf-8')

    def _validate_entry(self, entry, directory):
        index=len(self.entries)
        if type(entry.get('index')) is not int or entry['index']!=index or index>=len(self.records):
            raise ValueError('Resume entries must be a unique ordered source prefix')
        name=entry.get('file','')
        if not re.fullmatch(r'[0-9]{8}\.npy',name) or name!=f'{index:08d}.npy':
            raise ValueError('Unsafe or mismatched resume array filename')
        if entry.get('record')!=self.records[index]:raise ValueError('Resume source record changed')
        if sha(directory/name)!=entry.get('sha256'):raise ValueError('Changed encoder resume array')
        values=np.load(directory/name,allow_pickle=False)
        self._validate_values(values,self.records[index])
        receipt=entry.get('frame_receipt',{})
        record=self.records[index]
        if receipt.get('video')!=record['video'] or receipt.get('source_sha256')!=record['sha256'] or receipt.get('split')!=record['split']:
            raise ValueError('Resume frame/source identity changed')
        if not isinstance(receipt.get('samples'),list) or len(receipt['samples'])!=len(values):
            raise ValueError('Resume frame schedule length changed')
        from src.models.moment_encoder import frame_schedule
        if receipt['samples']!=frame_schedule(record['num_segments'],receipt.get('duration',0),receipt.get('fps',0)):
            raise ValueError('Resume frame schedule changed')

    @staticmethod
    def _validate_values(values, record):
        if values.dtype!=np.float32 or values.shape!=(2*record['num_segments'],512):
            raise ValueError('Resume feature shape or dtype changed')
        if not np.isfinite(values).all() or not np.allclose(np.linalg.norm(values,axis=1),1,atol=1e-4):
            raise ValueError('Nonfinite or unnormalized resume features')

    def get(self,index):
        if type(index) is not int or index<0 or index>=len(self.records):
            raise ValueError('Valid source index required')
        if index>=len(self.entries):return None
        entry=self.entries[index]
        return np.load(self.directory/entry['file'],allow_pickle=False),entry['frame_receipt']

    def append(self,index,values,receipt):
        if type(index) is not int or index!=len(self.entries) or index>=len(self.records):
            raise ValueError('Only the next source may be checkpointed')
        self._validate_values(values,self.records[index])
        name=f'{index:08d}.npy';buffer=io.BytesIO()
        np.save(buffer,values,allow_pickle=False)
        content=buffer.getvalue()
        with (self.directory/name).open('xb') as handle:handle.write(content)
        entry={'index':index,'file':name,'sha256':hashlib.sha256(content).hexdigest(),
               'record':self.records[index],'frame_receipt':receipt}
        self._validate_entry(entry,self.directory)
        # A crash can leave an orphan array or an incomplete final line. Resume
        # copies only verified complete entries into a fresh destination.
        with (self.directory/'complete.jsonl').open('a',encoding='utf-8',newline='\n') as handle:
            handle.write(json.dumps(entry,allow_nan=False)+'\n');handle.flush()
        self.entries.append(entry)
