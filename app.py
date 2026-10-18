import os, io, json, base64
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

ROOT=os.path.dirname(os.path.abspath(__file__))
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
app=FastAPI(title="AIShield Adversarial AI Security Lab")
app.mount("/static",StaticFiles(directory=os.path.join(ROOT,"static")),name="static")

class SmallCNN(nn.Module):
    def __init__(self,channels,classes):
        super().__init__()
        self.features=nn.Sequential(nn.Conv2d(channels,32,3,padding=1),nn.ReLU(),nn.Conv2d(32,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(64,96,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d((4,4)))
        self.classifier=nn.Sequential(nn.Flatten(),nn.Linear(96*4*4,128),nn.ReLU(),nn.Dropout(.2),nn.Linear(128,classes))
    def forward(self,x): return self.classifier(self.features(x))

CFG={"mnist":{"channels":1,"classes":10,"eps":.15,"files":["mnist_baseline.pt","mnist_adversarially_trained.pt"]},
     "gtsrb":{"channels":3,"classes":43,"eps":.03,"files":["gtsrb_baseline.pt","gtsrb_adversarially_trained.pt"]}}
GTSRB=["Speed limit 20","Speed limit 30","Speed limit 50","Speed limit 60","Speed limit 70","Speed limit 80","End speed limit 80","Speed limit 100","Speed limit 120","No passing","No passing >3.5 tons","Right-of-way at next intersection","Priority road","Yield","Stop","No vehicles","Vehicles >3.5 tons prohibited","No entry","General caution","Dangerous curve left","Dangerous curve right","Double curve","Bumpy road","Slippery road","Road narrows right","Road work","Traffic signals","Pedestrians","Children crossing","Bicycles crossing","Beware ice/snow","Wild animals crossing","End speed limit + passing limits","Turn right ahead","Turn left ahead","Ahead only","Straight or right","Straight or left","Keep right","Keep left","Roundabout mandatory","End no passing","End no passing >3.5 tons"]
LABELS={"mnist":[str(i) for i in range(10)],"gtsrb":GTSRB}
PRE={"mnist":transforms.Compose([transforms.Grayscale(1),transforms.Resize((32,32)),transforms.ToTensor()]),"gtsrb":transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor()])}
MODELS={}
def load_models():
    for task,cfg in CFG.items():
        MODELS[task]={}
        for variant,filename in zip(("baseline","defended"),cfg["files"]):
            p=os.path.join(ROOT,"models",filename); m=SmallCNN(cfg["channels"],cfg["classes"]).to(DEVICE)
            try:
                if os.path.exists(p): m.load_state_dict(torch.load(p,map_location=DEVICE,weights_only=True));m.eval();MODELS[task][variant]=m
                else: MODELS[task][variant]=None
            except Exception as e: print("Model load error",p,e);MODELS[task][variant]=None
load_models()

def require(task,variant):
    m=MODELS[task][variant]
    if m is None: raise HTTPException(503,f"Missing models/{CFG[task]['files'][0 if variant=='baseline' else 1]}. Run train_models.py first, then deploy the generated weights.")
    return m
def pred(m,x):
    with torch.no_grad():
        p=F.softmax(m(x),1)[0];i=int(p.argmax())
    return i,float(p[i]),p.cpu().numpy()
def attack_fgsm(m,x,y,eps):
    z=x.detach().clone().requires_grad_(True); loss=F.cross_entropy(m(z),y);g=torch.autograd.grad(loss,z)[0]
    return (z+eps*g.sign()).detach().clamp(0,1)
def attack_pgd(m,x,y,eps,steps):
    orig=x.detach();z=(orig+torch.empty_like(orig).uniform_(-eps,eps)).clamp(0,1)
    for _ in range(steps):
        z.requires_grad_(True);loss=F.cross_entropy(m(z),y);g=torch.autograd.grad(loss,z)[0]
        with torch.no_grad(): z=(z+(eps/4)*g.sign());z=torch.max(torch.min(z,orig+eps),orig-eps).clamp(0,1)
    return z.detach()
def dataurl(x):
    x=x.detach().cpu().squeeze(0).clamp(0,1)
    if x.shape[0]==1: im=Image.fromarray((x[0].numpy()*255).astype("uint8")).convert("RGB")
    else: im=Image.fromarray((x.permute(1,2,0).numpy()*255).astype("uint8"))
    b=io.BytesIO();im.save(b,format="PNG");return "data:image/png;base64,"+base64.b64encode(b.getvalue()).decode()

def difference_dataurl(original, changed):
    """Return a visible, contrast-enhanced image difference for teaching/demo use."""
    a = original.detach().cpu().squeeze(0).clamp(0, 1)
    b = changed.detach().cpu().squeeze(0).clamp(0, 1)
    diff = (a - b).abs().mean(dim=0).numpy()
    if diff.max() > 0:
        diff = diff / diff.max()
    # heatmap: red channel emphasizes changed pixels
    heat = np.zeros((diff.shape[0], diff.shape[1], 3), dtype=np.uint8)
    heat[..., 0] = (diff * 255).astype("uint8")
    heat[..., 1] = (np.sqrt(diff) * 90).astype("uint8")
    im = Image.fromarray(heat)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def benchmark_rows():
    p = os.path.join(ROOT, "results", "benchmark.json")
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return None

@app.get("/",response_class=HTMLResponse)
def home(): return open(os.path.join(ROOT,"static","index.html"),encoding="utf-8").read()
@app.get("/api/status")
def status(): return {"device":str(DEVICE),"models":{t:{v:MODELS[t][v] is not None for v in MODELS[t]} for t in MODELS}}
@app.get("/api/benchmark")
def benchmark():
    p=os.path.join(ROOT,"results","benchmark.json")
    if not os.path.exists(p): return {"available":False,"message":"No measured benchmark yet. Run train_models.py."}
    return {"available":True,"results":json.load(open(p,encoding="utf-8"))}
@app.post("/api/analyze")
async def analyze(image:UploadFile=File(...),task:str=Form("gtsrb"),variant:str=Form("baseline"),attack:str=Form("none"),epsilon:float=Form(.03),steps:int=Form(7)):
    if task not in CFG or variant not in ("baseline","defended") or attack not in ("none","fgsm","pgd"): raise HTTPException(400,"Invalid experiment settings")
    maxeps=.3 if task=="mnist" else .1
    if not 0<=epsilon<=maxeps or not 1<=steps<=30: raise HTTPException(400,"Perturbation or step count outside allowed demo range")
    m=require(task,variant)
    try: im=Image.open(io.BytesIO(await image.read())).convert("RGB");x=PRE[task](im).unsqueeze(0).to(DEVICE)
    except Exception as e: raise HTTPException(400,"Please upload a valid PNG or JPEG image.")
    ci,cc,_=pred(m,x);xa=x
    if attack!="none":
        # For unlabeled uploads, current clean prediction is used as a pseudo-label.
        y=torch.tensor([ci],device=DEVICE)
        xa=attack_fgsm(m,x,y,epsilon) if attack=="fgsm" else attack_pgd(m,x,y,epsilon,steps)
    ri,rc,probs=pred(m,xa)
    with torch.no_grad():
        p1=F.softmax(m(xa),1);p2=F.softmax(m(torch.round(xa*7)/7),1);score=float((p1-p2).abs().mean())
    top=np.argsort(probs)[-5:][::-1]
    return {"task":task,"variant":variant,"attack":attack,"epsilon":epsilon,"clean":{"id":ci,"label":LABELS[task][ci],"confidence":cc,"image":dataurl(x)},"result":{"id":ri,"label":LABELS[task][ri],"confidence":rc,"image":dataurl(xa)},"difference_image":difference_dataurl(x, xa),"changed":ci!=ri,"squeeze_score":score,"top5":[{"label":LABELS[task][int(i)],"confidence":float(probs[i])} for i in top],"note":"Uploaded image has no verified ground-truth label; prediction change is not dataset attack-success rate."}


@app.post("/api/compare")
async def compare_models(image: UploadFile = File(...), task: str = Form("gtsrb"),
                         attack: str = Form("fgsm"), epsilon: float = Form(.03),
                         steps: int = Form(7)):
    """Run the same uploaded image and attack settings against both model variants."""
    if task not in CFG or attack not in ("none", "fgsm", "pgd"):
        raise HTTPException(400, "Invalid task or attack")
    maxeps = .3 if task == "mnist" else .1
    if not 0 <= epsilon <= maxeps or not 1 <= steps <= 30:
        raise HTTPException(400, "Perturbation or step count outside allowed demo range")
    try:
        raw = await image.read()
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        x = PRE[task](im).unsqueeze(0).to(DEVICE)
    except Exception:
        raise HTTPException(400, "Please upload a valid PNG or JPEG image.")
    output = {}
    for variant in ("baseline", "defended"):
        m = require(task, variant)
        ci, cc, _ = pred(m, x)
        xa = x
        if attack != "none":
            y = torch.tensor([ci], device=DEVICE)  # pseudo-label for unlabeled upload
            xa = attack_fgsm(m, x, y, epsilon) if attack == "fgsm" else attack_pgd(m, x, y, epsilon, steps)
        ri, rc, probs = pred(m, xa)
        output[variant] = {
            "clean_label": LABELS[task][ci], "clean_confidence": cc,
            "result_label": LABELS[task][ri], "result_confidence": rc,
            "prediction_changed": ci != ri,
            "top5": [{"label": LABELS[task][int(i)], "confidence": float(probs[i])}
                     for i in np.argsort(probs)[-5:][::-1]]
        }
    return {"task": task, "attack": attack, "epsilon": epsilon, "steps": steps,
            "models": output,
            "note": "Same uploaded image, but each model generates its own attack using its clean prediction as a pseudo-label. This is an illustrative comparison, not a labelled benchmark."}

@app.get("/api/security-score")
def security_score():
    """Transparent demo score derived from available measured benchmark rows, never a certification."""
    data = benchmark_rows()
    if not data or not data.get("results"):
        return {"available": False, "message": "Run python train_models.py to calculate a score from measured benchmark results."}
    rows = data["results"]
    defended = [r for r in rows if r.get("model") == "defended"]
    if not defended:
        defended = [r for r in rows if r.get("model") == "adversarially_trained"]
    if not defended:
        return {"available": False, "message": "No defended-model benchmark rows found."}
    clean = float(np.mean([r.get("clean_accuracy", 0) for r in defended]))
    fgsm = float(np.mean([r.get("fgsm_accuracy", 0) for r in defended]))
    pgd = float(np.mean([r.get("pgd_accuracy", 0) for r in defended]))
    # Simple transparent composite for this educational dashboard, not an industry standard.
    score = round(100 * (0.2 * clean + 0.4 * fgsm + 0.4 * pgd))
    return {"available": True, "score": score, "label": "Experimental robustness score",
            "components": {"clean_accuracy": clean, "fgsm_accuracy": fgsm, "pgd_accuracy": pgd},
            "formula": "100 × (0.20 × clean accuracy + 0.40 × FGSM accuracy + 0.40 × PGD accuracy)",
            "warning": "Educational composite only. Not a certification, safety rating, or guarantee of robustness."}

@app.post("/api/report")
async def create_report(image: UploadFile = File(...), task: str = Form("gtsrb"),
                        variant: str = Form("baseline"), attack: str = Form("fgsm"),
                        epsilon: float = Form(.03), steps: int = Form(7)):
    """Create a simple PDF from a real inference run and include limitations."""
    if task not in CFG or variant not in ("baseline", "defended") or attack not in ("none", "fgsm", "pgd"):
        raise HTTPException(400, "Invalid experiment settings")
    maxeps = .3 if task == "mnist" else .1
    if not 0 <= epsilon <= maxeps or not 1 <= steps <= 30:
        raise HTTPException(400, "Perturbation or step count outside allowed demo range")
    try:
        im = Image.open(io.BytesIO(await image.read())).convert("RGB")
        x = PRE[task](im).unsqueeze(0).to(DEVICE)
    except Exception:
        raise HTTPException(400, "Please upload a valid PNG or JPEG image.")
    m = require(task, variant)
    ci, cc, _ = pred(m, x)
    xa = x
    if attack != "none":
        y = torch.tensor([ci], device=DEVICE)
        xa = attack_fgsm(m, x, y, epsilon) if attack == "fgsm" else attack_pgd(m, x, y, epsilon, steps)
    ri, rc, probs = pred(m, xa)
    pdf = io.BytesIO()
    c = canvas.Canvas(pdf, pagesize=letter)
    width, height = letter
    y = height - 55
    c.setFont("Helvetica-Bold", 20); c.drawString(48, y, "AIShield Security Test Report"); y -= 30
    c.setFont("Helvetica", 10); c.drawString(48, y, "Educational adversarial robustness experiment"); y -= 28
    lines = [
        f"Task: {task.upper()}",
        f"Model: {variant}",
        f"Attack: {attack.upper()}",
        f"Epsilon: {epsilon}",
        f"PGD steps: {steps}",
        f"Original prediction: {LABELS[task][ci]} (confidence {cc:.4f})",
        f"Result prediction: {LABELS[task][ri]} (confidence {rc:.4f})",
        f"Prediction changed: {'Yes' if ci != ri else 'No'}",
        "",
        "Interpretation:",
        "This uploaded image has no verified ground-truth label. A prediction change",
        "is not a measured dataset attack-success rate. Confidence is not a security guarantee.",
        "",
        "Limitations:",
        "- Results apply only to this uploaded image and selected settings.",
        "- The attack uses the clean model prediction as a pseudo-label.",
        "- This report is not a certification or a physical-world safety evaluation.",
        "- Validate defenses on labelled held-out data before drawing conclusions."
    ]
    c.setFont("Helvetica", 11)
    for line in lines:
        if y < 55:
            c.showPage(); y = height - 55; c.setFont("Helvetica", 11)
        c.drawString(48, y, line[:110]); y -= 19
    c.save(); pdf.seek(0)
    return StreamingResponse(pdf, media_type="application/pdf",
                             headers={"Content-Disposition": "attachment; filename=aishield_security_report.pdf"})
