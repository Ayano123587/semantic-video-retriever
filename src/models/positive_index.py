"""Exact consensus labels without a full query-by-candidate boolean matrix."""
import numpy as np
from src.models.moments import consensus_iou,interval


class PositiveIndex:
    def __init__(self,candidates,queries):
        self.candidate_count=len(candidates)
        groups={};keys=set()
        for index,candidate in enumerate(candidates):
            moment=interval(candidate['moment']);key=(candidate['video'],moment)
            if key in keys:raise ValueError('Duplicate candidate moment')
            keys.add(key);groups.setdefault(candidate['video'],[]).append((index,moment))
        self.indices=[];self.local_indices=[]
        cache={}
        for query in queries:
            references=tuple(interval(t) for t in query['times'])
            if not references:raise ValueError('Missing temporal references')
            video=query['video']
            if video not in groups:raise ValueError('Correct video absent from candidate corpus')
            local=groups[video]
            self.local_indices.append(np.asarray([i for i,_ in local],dtype=np.int64))
            if any(t not in {moment for _,moment in local} for t in references):
                raise ValueError('Human reference missing from candidate geometry')
            key=(video,references)
            if key not in cache:
                cache[key]=np.asarray([i for i,m in local if consensus_iou(m,references)>=.5],dtype=np.int64)
            self.indices.append(cache[key])
        self.valid=np.asarray([len(v)>0 for v in self.indices],dtype=bool)

    def mask(self,query_indices):
        return self._mask(query_indices,self.indices)

    def same_video_mask(self,query_indices):
        return self._mask(query_indices,self.local_indices)

    def _mask(self,query_indices,values):
        query_indices=list(query_indices)
        result=np.zeros((len(query_indices),self.candidate_count),dtype=bool)
        for row,index in enumerate(query_indices):
            if isinstance(index,(bool,np.bool_)) or not isinstance(index,(int,np.integer)) or not 0<=index<len(self.indices):
                raise ValueError('Query index outside cohort')
            result[row,values[index]]=True
        return result
