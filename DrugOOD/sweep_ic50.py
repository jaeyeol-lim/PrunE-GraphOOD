"""Run the documented 108-config PrunE grid with a fixed 2-layer GIN selector."""
import argparse,itertools,subprocess,sys
from pathlib import Path
def main():
    p=argparse.ArgumentParser();p.add_argument("--domain",default="assay",choices=("assay","scaffold","size"));p.add_argument("--subset",default="core",choices=("core","general","refined"));p.add_argument("--endpoint",choices=("ic50","ec50"),default="ic50");p.add_argument("--seed",type=int,default=1);p.add_argument("--data-root",type=Path);p.add_argument("--output-root",type=Path,default=Path(__file__).parent/"sweeps");p.add_argument("--device",default="auto");p.add_argument("--dry-run",action="store_true");a,extra=p.parse_known_args();script=Path(__file__).parent/"train_ic50.py"
    grid=itertools.product((.5,.75,.85),(50,70,90),(10,40),(.1,.01,.001))
    for eta,k,l1,l2 in grid:
        name=f"eta_{eta:g}_k_{k}_l1_{l1:g}_l2_{l2:g}_gin_2l";out=a.output_root/a.endpoint/a.domain/name/f"seed_{a.seed}";cmd=[sys.executable,str(script),"--domain",a.domain,"--subset",a.subset,"--endpoint",a.endpoint,"--seed",str(a.seed),"--device",a.device,"--edge-budget",str(eta),"--edge-prob-thres",str(k),"--lambda1",str(l1),"--lambda2",str(l2),"--selector","gin","--selector-layers","2","--output-dir",str(out)];cmd+=( ["--data-root",str(a.data_root)] if a.data_root else [])+extra;print(" ".join(cmd))
        if not a.dry_run:subprocess.run(cmd,check=True)
if __name__=="__main__":main()
