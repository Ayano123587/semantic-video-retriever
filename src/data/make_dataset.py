"""Fetch a 30-second CC-BY Blender sample from a pinned mirror."""
import argparse,hashlib,json,urllib.request
from pathlib import Path
REVISION='915c4b2aba75614b20dec3852375b394bb305f10'
URL=f'https://raw.githubusercontent.com/chintan9/Big-Buck-Bunny/{REVISION}/BigBuckBunny480p30s.mp4'
SHA256='84abe3ddf1938f0a6053cf6cc7874434177cdce927ba07d603cdd84c4c67a6ab'
def fetch(output):
    with urllib.request.urlopen(URL,timeout=60) as response:content=response.read(10_000_001)
    if hashlib.sha256(content).hexdigest()!=SHA256:raise ValueError('Pinned sample checksum mismatch')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);output.write_bytes(content)
    report={'url':URL,'revision':REVISION,'sha256':SHA256,'bytes':len(content),'license':'CC-BY-3.0','attribution':'(c) copyright 2008, Blender Foundation / www.bigbuckbunny.org','license_source':'https://peach.blender.org/about/','scope':'30-second low-resolution excerpt, not full movie'}
    Path('models').mkdir(exist_ok=True);Path('models/sample-provenance.json').write_text(json.dumps(report,indent=2));return report

EXTRA_SOURCES=[{'file':'elephants-dream.mp4','url':f'https://raw.githubusercontent.com/chintan9/Big-Buck-Bunny/{REVISION}/ElephantsDream.mp4','sha256':'5e77a77fbcb73ed2f1f2ff6b9361f2d9867de1390f5256ccfcce6ed755d9e704','license':'CC-BY-2.5','attribution':'(c) copyright 2006, Blender Foundation / Netherlands Media Art Institute / www.elephantsdream.org','license_source':'https://orange.blender.org/','scope':'37-second excerpt'}, {'file':'sintel-trailer.mp4','url':'https://media.w3.org/2010/05/sintel/trailer.mp4','sha256':'b670602fa00934ca27c4351bb0efe7ea7a07fae57284e44226025eeed7c51254','license':'CC-BY-3.0','attribution':'copyright Blender Foundation | www.sintel.org','license_source':'https://durian.blender.org/sharing/','scope':'52-second trailer'}]

def fetch_extra():
    for source in EXTRA_SOURCES:
        with urllib.request.urlopen(source['url'],timeout=60) as response:content=response.read(20_000_001)
        if hashlib.sha256(content).hexdigest()!=source['sha256']:raise ValueError('Corpus sample checksum mismatch')
        Path('data/raw',source['file']).write_bytes(content)
    Path('models/corpus-provenance.json').write_text(json.dumps(EXTRA_SOURCES,indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('data/raw/big-buck-bunny.mp4'));p.add_argument('--all',action='store_true',help='Also fetch validation and held-out movie excerpts');a=p.parse_args();print(json.dumps(fetch(a.output),indent=2))
    if a.all:fetch_extra()
if __name__=='__main__':main()
