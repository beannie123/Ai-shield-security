import os,json,random,time
import numpy as np
import torch,torch.nn as nn,torch.nn.functional as F
from torch.utils.data import DataLoader,Subset
from torchvision import datasets,transforms
from app import SmallCNN, attack_fgsm, attack_pgd, DEVICE, ROOT

SEED=42;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
EPOCHS=int(os.getenv("EPOCHS","3"));ADV_EPOCHS=int(os.getenv("ADV_EPOCHS","2"))
TRAIN_LIMIT=int(os.getenv("TRAIN_LIMIT","12000"));TEST_LIMIT=int(os.getenv("TEST_LIMIT","2000"));BATCH=128
os.makedirs(os.path.join(ROOT,"models"),exist_ok=True);os.makedirs(os.path.join(ROOT,"results"),exist_ok=True)
def loader(ds,limit,shuffle,seed):
    g=torch.Generator().manual_seed(seed);idx=torch.randperm(len(ds),generator=g)[:min(limit,len(ds))].tolist()
    return DataLoader(Subset(ds,idx),batch_size=BATCH,shuffle=shuffle,num_workers=2 if DEVICE.type=="cuda" else 0)
def train(m,dl,epochs,eps=None):
    opt=torch.optim.Adam(m.parameters(),lr=1e-3 if eps is None else 5e-4);start=time.time()
    for ep in range(epochs):
        m.train();tot=loss_sum=0
        for x,y in dl:
            x,y=x.to(DEVICE),y.to(DEVICE)
            if eps is not None:
                m.eval();xa=attack_fgsm(m,x,y,eps);m.train()
                loss=.5*F.cross_entropy(m(x),y)+.5*F.cross_entropy(m(xa),y)
            else: loss=F.cross_entropy(m(x),y)
            opt.zero_grad(set_to_none=True);loss.backward();opt.step();tot+=len(y);loss_sum+=loss.item()*len(y)
        print(f"epoch {ep+1}/{epochs}: loss={loss_sum/max(tot,1):.4f}")
    return time.time()-start
def evaluate(m,dl,eps):
    m.eval();tot=clean=fg=pg=elig=fs=ps=0
    for x,y in dl:
        x,y=x.to(DEVICE),y.to(DEVICE)
        with torch.no_grad():p=m(x).argmax(1)
        xa=attack_fgsm(m,x,y,eps);xp=attack_pgd(m,x,y,eps,5)
        with torch.no_grad():a=m(xa).argmax(1);b=m(xp).argmax(1)
        ok=p.eq(y);tot+=len(y);clean+=ok.sum().item();fg+=a.eq(y).sum().item();pg+=b.eq(y).sum().item()
        elig+=ok.sum().item();fs+=(ok&a.ne(y)).sum().item();ps+=(ok&b.ne(y)).sum().item()
    return {"clean_accuracy":clean/max(tot,1),"fgsm_accuracy":fg/max(tot,1),"pgd_accuracy":pg/max(tot,1),"fgsm_attack_success_rate":fs/max(elig,1),"pgd_attack_success_rate":ps/max(elig,1),"test_samples":tot}
def run(task,trds,teds,ch,classes,eps):
    tr=loader(trds,TRAIN_LIMIT,True,SEED);te=loader(teds,TEST_LIMIT,False,SEED+1)
    base=SmallCNN(ch,classes).to(DEVICE);print(f"\n{task}: train baseline")
    bt=train(base,tr,EPOCHS);torch.save(base.state_dict(),f"models/{task}_baseline.pt")
    defended=SmallCNN(ch,classes).to(DEVICE);defended.load_state_dict(base.state_dict());print(f"{task}: adversarial training")
    dt=train(defended,tr,ADV_EPOCHS,eps);torch.save(defended.state_dict(),f"models/{task}_adversarially_trained.pt")
    rows=[]
    for name,m in [("baseline",base),("defended",defended)]:
        rows.append({"module":task,"model":name,"epsilon":eps,"baseline_train_seconds":bt,"defended_train_seconds":dt,**evaluate(m,te,eps)})
    return rows
tf=transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor()])
mntr=datasets.MNIST("data",train=True,download=True,transform=tf);mnte=datasets.MNIST("data",train=False,download=True,transform=tf)
gttr=datasets.GTSRB("data",split="train",download=True,transform=tf);gtte=datasets.GTSRB("data",split="test",download=True,transform=tf)
rows=run("mnist",mntr,mnte,1,10,.15)+run("gtsrb",gttr,gtte,3,43,.03)
out={"device":str(DEVICE),"epochs":EPOCHS,"adversarial_epochs":ADV_EPOCHS,"results":rows,"warning":"Measured on selected subsets; not a certified robustness or physical-world safety evaluation."}
json.dump(out,open("results/benchmark.json","w"),indent=2);print(json.dumps(out,indent=2))
