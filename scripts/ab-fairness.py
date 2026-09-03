# Per-stream fairness view of bench-miaai (c=4, p=8192, 3 trials): prints each stream's TTFT and decode, plus min/max spread.
import asyncio, sys, statistics
src = open('/home/mia/models/deepSeek-v4-Flash-DSpark/scripts/bench-miaai.py').read().replace('asyncio.run(main())', '')
g = {}; exec(compile(src, 'bench-miaai.py', 'exec'), g)
BASE='http://127.0.0.1:8888/v1'; MODEL='deepseek-v4-flash-vision-exp'
P=int(sys.argv[1]) if len(sys.argv)>1 else 8192; C=int(sys.argv[2]) if len(sys.argv)>2 else 4; R=int(sys.argv[3]) if len(sys.argv)>3 else 3
label = sys.argv[4] if len(sys.argv)>4 else 'x'
async def main():
    tt=[]; dd=[]
    for rep in range(R):
        case = await g['run_case'](BASE, MODEL, P, C, f'fair-{label}-{rep}')
        t=[r['ttft_s'] for r in case['requests']]; d=[r['output_tok_s'] for r in case['requests']]
        tt.append(max(t)-min(t)); dd.append(max(d)-min(d))
        print(f"trial {rep}: c={C} p={P} agg={case['aggregate_tok_s']:.1f} ttft_per_stream={[round(x,2) for x in t]} s spread={max(t)-min(t):.2f} s  decode_per_stream={[round(x,1) for x in d]} spread={max(d)-min(d):.1f}", flush=True)
    print(f"FAIRNESS: ttft spread median {statistics.median(tt):.2f} s (min {min(tt):.2f} max {max(tt):.2f}); decode spread median {statistics.median(dd):.1f} tok/s (min {min(dd):.1f} max {max(dd):.1f})")
asyncio.run(main())
