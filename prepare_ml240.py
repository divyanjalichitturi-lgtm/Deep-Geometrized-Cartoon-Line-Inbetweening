import argparse, json, random, re
from pathlib import Path
import cv2
import numpy as np
from src.geometry import load_geometry_json, _scale_vertices

EXT={'.png','.jpg','.jpeg','.bmp'}
PROJECT_ROOT=Path(__file__).resolve().parent

def frame_no(p):
    m=re.findall(r'(\d+)',p.stem); return int(m[-1]) if m else -1

def find_split_roots(source):
    source=Path(source)
    candidates=[]
    for p in [source]+list(source.rglob('*')):
        if not p.is_dir(): continue
        if (p/'frames').is_dir() and (p/'labels').is_dir(): candidates.append(p)
    return candidates

def collect_sequences(source):
    seqs=[]
    # Prefer explicit ML240 layouts.
    for root in find_split_roots(source):
        froot=root/'frames'; lroot=root/'labels'
        for seq in sorted([p for p in froot.iterdir() if p.is_dir()]):
            label=lroot/seq.name
            if not label.is_dir(): continue
            frames=sorted([p for p in seq.iterdir() if p.suffix.lower() in EXT], key=frame_no)
            labels=list(label.glob('*.json'))
            if len(frames)>=3 and labels: seqs.append((seq,label))
    unique={str(s[0]):s for s in seqs}
    return list(unique.values())

def label_for(frame,label_dir):
    n=frame_no(frame); c=list(label_dir.glob(f'*{n:04d}*.json'))+list(label_dir.glob(f'*{n}*.json'))
    return sorted(set(c))[0] if c else None

def process_sequence(frame_dir,label_dir,out_dir,size):
    out_frames=out_dir/'frames'; out_labels=out_dir/'labels'; out_frames.mkdir(parents=True,exist_ok=True); out_labels.mkdir(parents=True,exist_ok=True)
    for frame in sorted([p for p in frame_dir.iterdir() if p.suffix.lower() in EXT],key=frame_no):
        lab=label_for(frame,label_dir)
        if lab is None: continue
        img=cv2.imread(str(frame),cv2.IMREAD_COLOR)
        if img is None: continue
        h,w=img.shape[:2]; resized=cv2.resize(img,size,interpolation=cv2.INTER_AREA)
        out_name=f'Image{frame_no(frame):04d}.png'; cv2.imwrite(str(out_frames/out_name),resized)
        verts, conns=load_geometry_json(str(lab)); verts=_scale_vertices(verts,w,h,size[0],size[1])
        # Preserve the discovered schema while replacing the vertex array where possible is hard; instead save a normalized geometry file.
        norm={"vertices":verts.tolist(),"connections":conns,"source_frame":frame.name,"source_label":lab.name,"source_size":[w,h],"target_size":list(size)}
        with open(out_labels/f'Line{frame_no(frame):04d}.json','w',encoding='utf-8') as f: json.dump(norm,f)

def main():
    ap=argparse.ArgumentParser(description='Prepare an ML240 subset using sequence-level, non-overlapping splits.')
    ap.add_argument('--source',required=True); ap.add_argument('--output',type=Path,default=PROJECT_ROOT/'dataset')
    ap.add_argument('--train',type=int,default=10); ap.add_argument('--val',type=int,default=2); ap.add_argument('--test',type=int,default=3)
    ap.add_argument('--size',type=int,default=256); ap.add_argument('--seed',type=int,default=42)
    a=ap.parse_args(); random.seed(a.seed)
    if min(a.train,a.val,a.test)<0 or a.train+a.val+a.test<3: raise SystemExit('Split counts must be non-negative and select at least three sequences.')
    if a.size<32: raise SystemExit('--size must be at least 32 pixels.')
    seqs=collect_sequences(a.source)
    if len(seqs)<a.train+a.val+a.test: raise RuntimeError(f'Found only {len(seqs)} usable sequences; need {a.train+a.val+a.test}.')
    random.shuffle(seqs); groups=[('train',a.train),('val',a.val),('test',a.test)]; idx=0
    out=a.output if a.output.is_absolute() else PROJECT_ROOT/a.output
    if any((out/s).exists() and any((out/s).iterdir()) for s in ('train','val','test')):
        raise SystemExit(f'Prepared data already exists at {out}. Use a new --output directory to preserve the existing split.')
    out.mkdir(parents=True,exist_ok=True)
    for split,n in groups:
        for j in range(n):
            f,l=seqs[idx]; idx+=1; name=f'{j+1:03d}_{f.name}'
            process_sequence(f,l,out/split/name,(a.size,a.size))
    print(f'Prepared {a.train} train + {a.val} val + {a.test} test sequences at {out.resolve()} (sequence-level disjoint split, seed={a.seed})')

if __name__=='__main__': main()
