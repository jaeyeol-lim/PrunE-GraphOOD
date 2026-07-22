"""Run the official PrunE implementation on cached DrugOOD IC50 splits."""
from __future__ import annotations
import argparse,json,random,sys,time
from pathlib import Path
import numpy as np
import torch
from torch_geometric.loader import DataLoader

REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
try:
    from .data import discover_data_root,load_splits
except ImportError:
    from data import discover_data_root,load_splits
from modelNew.prune import Model

def parse_args():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--domain",choices=("assay","scaffold","size"),default="assay");p.add_argument("--subset",choices=("core","general","refined"),default="core");p.add_argument("--endpoint",choices=("ic50","ec50"),default="ic50");p.add_argument("--data-root",type=Path,default=discover_data_root());p.add_argument("--output-dir",type=Path);p.add_argument("--device",default="cuda:0");p.add_argument("--seed",type=int,default=1)
    p.add_argument("--epochs",type=int,default=50);p.add_argument("--erm-pretrain-epochs",type=int,default=10);p.add_argument("--patience",type=int,default=10);p.add_argument("--batch-size",type=int,default=128);p.add_argument("--num-workers",type=int,default=4);p.add_argument("--lr",type=float,default=1e-3);p.add_argument("--weight-decay",type=float,default=0.);p.add_argument("--hidden-dim",type=int,default=128);p.add_argument("--num-layers",type=int,default=4);p.add_argument("--dropout",type=float,default=.1)
    p.add_argument("--edge-budget",type=float,default=.75);p.add_argument("--edge-prob-thres",type=int,default=70);p.add_argument("--lambda1",type=float,default=10.);p.add_argument("--lambda2",type=float,default=.01);p.add_argument("--selector",choices=("gin",),default="gin");p.add_argument("--selector-layers",type=int,choices=(2,),default=2);p.add_argument("--selection-metric",choices=("accuracy","roc_auc"),default="accuracy");return p.parse_args()
def seed_all(s):
    random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
def main(a):
    seed_all(a.seed);dev=torch.device(a.device if a.device!="auto" else ("cuda" if torch.cuda.is_available() else "cpu"));stem,splits=load_splits(a.data_root,a.subset,a.domain,a.endpoint);pin=dev.type=="cuda";loader=lambda ds,shuffle:DataLoader(ds,batch_size=a.batch_size,shuffle=shuffle,num_workers=a.num_workers,pin_memory=pin,persistent_workers=a.num_workers>0);tr=loader(splits["train"],True);val=loader(splits["ood_val"],False);test=loader(splits["ood_test"],False);sample=splits["train"][0]
    kwargs=dict(nfeat=sample.x.shape[-1],nhid=a.hidden_dim,nclass=2,nlayers=a.num_layers,dropout=a.dropout,edge_dim=sample.edge_attr.shape[-1],jk="last",node_cls=False,pooling="sum",with_bn=True,weight_decay=a.weight_decay,lr=a.lr,adapt_lr=1e-4,lr_scheduler=True,patience=50,early_stop_epochs=a.patience,lr_decay=.75,project_layer_num=2,edge_gnn=a.selector,edge_gnn_layers=a.selector_layers,edge_budget=a.edge_budget,num_samples=1,edge_penalty=a.lambda1,edge_uniform_penalty=a.lambda2,edge_prob_thres=a.edge_prob_thres,featureMasking=False,temp=.2,adapt_params="edge_gnn",base_gnn="gin",valid_metric="acc" if a.selection_metric=="accuracy" else "auc",device=dev,debug=False,useAutoAug=True,dataset_name=stem,mol_encoder=False,use_vGIN=False,use_rsc=False,use_div_cls=False,uniform_p=-1.,pretraining_epochs=a.erm_pretrain_epochs)
    model=Model(**kwargs).to(dev)
    # The released Model references this diagnostic flag without initializing it.
    # Defining it externally only disables optional ground-truth-edge diagnostics.
    model.calc_edge_stats=False
    model.fit(tr,val,test,epochs=a.erm_pretrain_epochs+a.epochs)
    if model.best_states is None:raise RuntimeError("Official PrunE training produced no validation checkpoint")
    model.load_state_dict(model.best_states);pairs=sorted(model.valid_metric_list,key=lambda x:x[0],reverse=True);best_val,best_test=pairs[0];out=a.output_dir or Path(__file__).resolve().parent/"outputs"/f"prune_{stem}_seed{a.seed}_{int(time.time())}";out.mkdir(parents=True,exist_ok=True);torch.save({"model":model.state_dict(),"args":vars(a)},out/"best.pt");summary={"method":"PrunE","dataset":stem,"seed":a.seed,"best_ood_val":best_val,"metrics":{"ood_test":{a.selection_metric:best_test}},"args":{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}};(out/"summary.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=="__main__":main(parse_args())
