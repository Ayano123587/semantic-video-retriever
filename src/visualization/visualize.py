"""Generate an EDA contact sheet for manual timestamp annotation."""
import argparse
from PIL import Image,ImageDraw
from src.features.build_features import sample

def main():
    p=argparse.ArgumentParser();p.add_argument('video');p.add_argument('--output',default='data/interim/contact.png');a=p.parse_args();frames=list(sample(a.video));canvas=Image.new('RGB',(600,130*((len(frames)+2)//3)));draw=ImageDraw.Draw(canvas)
    for i,(frame,r) in enumerate(frames):
        frame.thumbnail((200,105));x=(i%3)*200;y=(i//3)*130;canvas.paste(frame,(x,y));draw.text((x,y+105),f"{r['timestamp_seconds']:.2f}s")
    canvas.save(a.output)
if __name__=='__main__':main()
