"""Temporal retrieval metrics for inclusive DiDeMo five-second bin intervals.

Corpus consensus recall is a custom metric. Official within-video metrics follow
LisaAnne/LocalizingMoments utils/eval.py at the pinned data revision.
"""
import numpy as np


def interval(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError('Expected inclusive start/end bins')
    a,b=value
    if type(a) is not int or type(b) is not int or not 0 <= a <= b < 6:
        raise ValueError('Invalid DiDeMo bin interval')
    return a,b


def moments(num_segments):
    if type(num_segments) is not int or not 1 <= num_segments <= 6:
        raise ValueError('Expected one through six segments')
    return [(a,b) for a in range(num_segments) for b in range(a,num_segments)]


def iou(prediction, reference):
    a,b=interval(prediction);c,d=interval(reference)
    return max(0,min(b,d)+1-max(a,c))/(max(b,d)+1-min(a,c))


def consensus_iou(prediction, references):
    if not references:raise ValueError('Missing human temporal references')
    values=sorted(iou(prediction,ref) for ref in references)
    return float(np.mean(values[-3:]))


def positive_mask(candidates, query):
    return np.asarray([c['video']==query['video'] and
        consensus_iou(c['moment'],query['times']) >= .5 for c in candidates],dtype=bool)


def evaluate(scores, candidates, queries):
    scores=np.asarray(scores)
    if scores.shape != (len(queries),len(candidates)) or not queries or not candidates:
        raise ValueError('Nonempty query by candidate score matrix required')
    if not np.isfinite(scores).all():raise ValueError('Nonfinite retrieval scores')
    keys=[(c['video'],interval(c['moment'])) for c in candidates]
    if len(set(keys))!=len(keys):raise ValueError('Duplicate candidate moments')
    corpus1=[];corpus5=[];video1=[];video5=[];local1=[];local5=[];local_iou=[];attainable=[]
    for row,q in zip(scores,queries):
        refs=[interval(t) for t in q['times']]
        if not refs:raise ValueError('Missing temporal references')
        order=np.argsort(-row,kind='stable')
        local=[i for i in order if candidates[i]['video']==q['video']]
        if not local:raise ValueError('Correct video absent from candidate corpus')
        local_moments=[keys[i][1] for i in local]
        if any(t not in local_moments for t in refs):
            raise ValueError('Human reference missing from candidate geometry')
        positives=positive_mask(candidates,q)
        corpus1.append(bool(positives[order[0]]));corpus5.append(bool(positives[order[:5]].any()))
        video1.append(candidates[order[0]]['video']==q['video'])
        video5.append(any(candidates[i]['video']==q['video'] for i in order[:5]))
        ranks=sorted(local_moments.index(t)+1 for t in refs)
        average_rank=float(np.mean(ranks[:3]))
        local1.append(average_rank <= 1);local5.append(average_rank <= 5)
        local_iou.append(consensus_iou(local_moments[0],refs));attainable.append(bool(positives.any()))
    mean=lambda x:float(np.mean(x))
    return {'queries':len(queries),'candidate_moments':len(candidates),
        'corpus_consensus_r1_iou50':mean(corpus1),'corpus_consensus_r5_iou50':mean(corpus5),
        'corpus_video_r1':mean(video1),'corpus_video_r5':mean(video5),
        'within_video_official_rank1':mean(local1),'within_video_official_rank5':mean(local5),
        'within_video_official_miou':mean(local_iou),
        'consensus_attainable_fraction':mean(attainable),
        'metric_scope':'licensed available subset; corpus metrics custom; within-video formula official',
        'tie_rule':'stable original candidate order; no gold-dependent tie breaking'}
