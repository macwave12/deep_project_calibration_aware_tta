# Run from the repo root: python3 scripts/check_temperature_near_ties.py 8100
"""On the 8,100-step stream: at every step whose exact gaps have a runner-up within
WIDE of the best (a superset of every step where the reference could differ), run the
full reference search and compare. Also reports the pure-exact (tol=None) result."""
import os, sys, time, warnings
sys.path.insert(0, os.getcwd())
warnings.filterwarnings("ignore")
import torch
torch.Tensor.cuda = lambda self, *a, **k: self
import yaml
from online_method.cal_tda import CalTDA, CalTdaConfig
from tests.cpu_stream import FakeCLIP, default_args, feature_stream
from utils.loo_temperature import search_temperature

WIDE = 1e-4
log = []

class Checked(CalTDA):
    def _search_temperature(self, estimate):
        t_new = super()._search_temperature(estimate)
        conf = self._confidence
        if estimate is None or estimate != estimate:
            return t_new
        gaps = (conf.mean_confidences() - estimate).abs()
        best = int(torch.argmin(gaps))
        t_pure = conf.grid[best]
        srt = gaps.sort().values
        if float(srt[1] - srt[0]) <= WIDE:
            t_ref = search_temperature(torch.cat(conf._history), estimate)
            log.append((conf.count, t_new, t_pure, t_ref, float(srt[1] - srt[0])))
        return t_new

n = int(sys.argv[1]) if len(sys.argv) > 1 else 8100
f, t, _ = feature_stream(n, 10, 512, 0)
cfg = CalTdaConfig.from_dict(yaml.safe_load(open("configs/method/cal_tda_temp.yaml")))
r = Checked(FakeCLIP(f, t), None, config=cfg)
args = default_args("eurosat")
r.prepare_model_and_optimization(args)
s = time.perf_counter()
for _ in range(n):
    r.adaptation_process(None, torch.zeros(1), args)
print(f"run {time.perf_counter()-s:.1f}s; rescored (tol 1e-5) {r._confidence.n_rescored} of {n} searches")
print(f"steps checked against full reference (runner-up within {WIDE}): {len(log)}")
print("new != ref:", [x for x in log if x[1] != x[3]])
print("pure-exact != ref:", [x for x in log if x[2] != x[3]])
