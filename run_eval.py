import subprocess
import numpy as np

for midx in (np.arange(68)+1)*1000:
# midx = 1000
    mdir = f"./trained_model_test/checkpoint-{midx}"
    print(mdir)
    o = subprocess.run(f"python3 run_denoise.py --do_eval --task nli --dataset snli --model {mdir} --output_dir ./eval_output_test/checkpoint-{midx}", shell=True, check=True)

# i luv u