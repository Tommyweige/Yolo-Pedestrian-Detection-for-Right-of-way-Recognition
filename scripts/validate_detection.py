"""Exercise the production bridge/backend with actual weights, then decode its output."""
import argparse
import json
import time
from pathlib import Path
import sys

import cv2

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from detection_backend import run_detection

parser=argparse.ArgumentParser()
parser.add_argument('--video',required=True)
parser.add_argument('--output',required=True)
parser.add_argument('--model',choices=['yolov8s','yolov8l','yolov8x6'],default='yolov8s')
parser.add_argument('--task',choices=['tf','zebra'],default='zebra')
parser.add_argument('--verify-only',action='store_true',help='Validate a completed run from its event log')
parser.add_argument('--gui-report',help='Validate a completed Rust screenshot check and its output')
args=parser.parse_args()
output=Path(args.output).resolve(); output.mkdir(parents=True,exist_ok=True)
input_capture=cv2.VideoCapture(args.video)
assert input_capture.isOpened()
input_frames=int(input_capture.get(cv2.CAP_PROP_FRAME_COUNT)); input_fps=input_capture.get(cv2.CAP_PROP_FPS)
input_capture.release()
started=time.perf_counter(); progress=[]
try:
    if args.gui_report:
        report=json.loads(Path(args.gui_report).read_text(encoding='utf-8'))
        assert report['error'] is None and report['batch_progress']==report['single_progress']==1.0,report
        event={'type':'done'}
        elapsed=report['elapsed_seconds']
    elif args.verify_only:
        events=[json.loads(line) for line in (output/'events.jsonl').read_text(encoding='utf-8').splitlines()]
        event=events[-1]
        progress=[item for item in events if item['type']=='progress' and item['total']>1]
        elapsed=event['elapsed_seconds']
    else:
        with (output/'events.jsonl').open('w',encoding='utf-8') as events_file:
            for event in run_detection([args.video],str(output),args.model,args.task):
                event['elapsed_seconds']=time.perf_counter()-started
                events_file.write(json.dumps(event,ensure_ascii=False)+'\n'); events_file.flush()
                if event['type']=='progress' and event['total']>1:
                    progress.append(event)
                    if event['current']%60==0 or event['current']==event['total']:
                        print(json.dumps({'frame':event['current'],'total':event['total'],'seconds':round(event['elapsed_seconds'],2)}),flush=True)
        elapsed=time.perf_counter()-started
    assert event['type']=='done'
    if not args.gui_report:
        assert progress and progress[-1]['current']==input_frames
    videos=[path for path in output.rglob('*.mp4') if path.is_file()]
    assert len(videos)==1, f'Expected one output video: {videos}'
    capture=cv2.VideoCapture(str(videos[0])); assert capture.isOpened()
    output_fps=capture.get(cv2.CAP_PROP_FPS); decoded=0
    while True:
        ok,frame=capture.read()
        if not ok: break
        decoded+=1
    capture.release()
    assert decoded==input_frames,(decoded,input_frames)
    assert abs(output_fps-input_fps)<0.01,(output_fps,input_fps)
    summary={'status':'pass','model':args.model,'task':args.task,'input':str(Path(args.video).resolve()),
             'validation_route':'rust-gui' if args.gui_report else 'backend',
             'input_frames':input_frames,'input_fps':input_fps,'output_video':str(videos[0]),
             'decoded_output_frames':decoded,'output_fps':output_fps,'elapsed_seconds':elapsed,
             'end_to_end_fps':input_frames/elapsed}
except Exception as error:
    summary={'status':'fail','model':args.model,'task':args.task,'error':str(error),'elapsed_seconds':time.perf_counter()-started}
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    raise
(output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=True),flush=True)
