"""The Kaggle session's own deadline, put in front of a kernel's code: a stuck stage
writes report.json with the reason and ends the session instead of burning GPU hours."""

WATCHDOG = """
# ---- the session's own deadline (as in the shared kaggle/job_template.py) ----
import threading as _th, os as _os, json as _js, time as _tm, pathlib as _pl, sys as _sys
_OUT = _pl.Path("/kaggle/working"); _OUT.mkdir(parents=True, exist_ok=True)
_BUDGET_S = float(JOB.get("budget_s") or 900)
_STAGE = {"name": "starting", "since": _tm.time(), "began": _tm.time()}
def _mark(name):
    _STAGE["name"], _STAGE["since"] = name, _tm.time()
    print(f"-- {name} (at {_tm.time() - _STAGE['began']:.0f}s)", flush=True)
def _give_up():
    ran = _tm.time() - _STAGE["began"]
    rep = {"errors": [f"timeout: {ran:.0f}s over a {2 * _BUDGET_S:.0f}s deadline, stuck in {_STAGE['name']}"],
           "timed_out": {"ran_s": round(ran, 1), "stage": _STAGE["name"]}}
    try:
        (_OUT / "report.json").write_text(_js.dumps(rep, indent=2))
    except Exception:
        pass
    print("==== GAVE UP ====", rep, flush=True)
    _os._exit(0)
_WD = _th.Timer(2 * _BUDGET_S, _give_up); _WD.daemon = True; _WD.start()
def _need_gpu():
    import torch
    if not torch.cuda.is_available():
        (_OUT / "report.json").write_text(_js.dumps({"errors": ["no GPU in this session"]}))
        print("no GPU in this session; stopping before loading anything", flush=True)
        _os._exit(0)
"""
