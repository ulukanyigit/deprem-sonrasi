
import sys, runpy, traceback, os, time
tgt = sys.argv[1]; logf = os.path.join(os.environ.get("DEPREM_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))), "data", "il", "calistir_hata.txt")
try:
    sys.argv = sys.argv[1:]; runpy.run_path(tgt, run_name="__main__")
    open(logf,"a",encoding="utf-8").write(time.strftime("[%H:%M:%S] ")+tgt+" OK\n")
except BaseException:
    open(logf,"a",encoding="utf-8").write(time.strftime("[%H:%M:%S] ")+tgt+" HATA\n"+traceback.format_exc()+"\n")
