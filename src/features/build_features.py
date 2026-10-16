"""Timestamped frame sampling with a lightweight sampled-shot-change heuristic."""
import math
import imageio.v2 as imageio
import numpy as np
from PIL import Image

def frame_schedule(fps,duration,interval):
    if not all(math.isfinite(v) and v>0 for v in [fps,duration,interval]):raise ValueError('fps, duration and interval must be positive finite')
    indices=sorted(set(round(t*fps) for t in np.arange(0,duration,interval)))
    return [(i,i/fps) for i in indices if i/fps<duration]

def sample(video,interval=2.,cut_threshold=.18):
    if not 0<cut_threshold<=1:raise ValueError('cut_threshold must be in (0,1]')
    reader=imageio.get_reader(str(video));previous=None;scene=0
    try:
        metadata=reader.get_meta_data()
        # Container duration may include audio past the last video frame.
        duration=min(metadata['duration'],reader.count_frames()/metadata['fps'])
        for number,timestamp in frame_schedule(metadata['fps'],duration,interval):
            frame=Image.fromarray(reader.get_data(number)).convert('RGB')
            small=np.asarray(frame.resize((32,32)),dtype=np.float32)/255
            if previous is not None and np.abs(small-previous).mean()>cut_threshold:scene+=1
            previous=small
            yield frame,{'frame_number':number,'timestamp_seconds':timestamp,'scene_id':scene}
    finally:reader.close()
