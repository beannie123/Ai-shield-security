const $=id=>document.getElementById(id);let task="gtsrb",file=null,chart=null;
async function req(url,opts={}){let r=await fetch(url,opts),d=await r.json();if(!r.ok)throw Error(d.detail||"Request failed");return d}
function esc(s){return String(s).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]))}
function toast(s){$("toast").textContent=s;$("toast").classList.remove("hidden");setTimeout(()=>$("toast").classList.add("hidden"),5000)}
document.querySelectorAll(".nav").forEach(b=>b.onclick=()=>{document.querySelectorAll(".nav").forEach(n=>n.classList.toggle("active",n===b));["lab","benchmark","method"].forEach(v=>$("view-"+v).classList.toggle("hidden",v!==b.dataset.view));$("title").textContent=b.textContent.trim();if(b.dataset.view==="benchmark")bench()});
document.querySelectorAll("[data-task]").forEach(b=>b.onclick=()=>{task=b.dataset.task;document.querySelectorAll("[data-task]").forEach(x=>x.classList.toggle("active",x===b));$("eps").max=task==="mnist"?.3:.1;$("eps").step=task==="mnist"?.01:.005;$("eps").value=task==="mnist"?.15:.03;$("epsval").textContent=Number($("eps").value).toFixed(3)});
document.querySelectorAll('input[name="attack"]').forEach(x=>x.onchange=()=> $("pgdrow").classList.toggle("hidden",document.querySelector('input[name="attack"]:checked').value!=="pgd"));
$("eps").oninput=()=> $("epsval").textContent=Number($("eps").value).toFixed(3);$("steps").oninput=()=> $("stepsval").textContent=$("steps").value;
function setfile(f){if(!f)return;if(!f.type.startsWith("image/")){toast("Please select an image file.");return}file=f;$("filename").textContent=f.name;$("cleanimg").innerHTML=`<img src="${URL.createObjectURL(f)}" alt="selected image">`}
$("file").onchange=e=>setfile(e.target.files[0]);$("upload").ondragover=e=>e.preventDefault();$("upload").ondrop=e=>{e.preventDefault();setfile(e.dataTransfer.files[0])};
$("run").onclick=async()=>{if(!file){toast("Choose an image first.");return}let attack=document.querySelector('input[name="attack"]:checked').value;let f=new FormData();f.append("image",file);f.append("task",task);f.append("variant",$("variant").value);f.append("attack",attack);f.append("epsilon",$("eps").value);f.append("steps",$("steps").value);$("run").disabled=true;$("run").textContent="Running model…";try{let d=await req("/api/analyze",{method:"POST",body:f});$("cleanimg").innerHTML=`<img src="${d.clean.image}" alt="original">`;$("resultimg").innerHTML=`<img src="${d.result.image}" alt="result">`;$("cleanlabel").textContent=d.clean.label+" (#"+d.clean.id+")";$("resultlabel").textContent=d.result.label+" (#"+d.result.id+")";$("cleanconf").textContent="Confidence "+(100*d.clean.confidence).toFixed(2)+"%";$("resultconf").textContent="Confidence "+(100*d.result.confidence).toFixed(2)+"%";$("runstatus").textContent=d.changed?"Prediction changed":"Prediction unchanged";$("resulttitle").textContent=attack==="none"?"MODEL INPUT":attack.toUpperCase()+" OUTPUT";$("callout").textContent=d.note;$("heatmap").innerHTML=`<img src="${d.difference_image}" alt="Difference heatmap; brighter red pixels indicate larger changes">`;$("changed").textContent=d.changed?"YES":"NO";$("cc").textContent=(100*d.clean.confidence).toFixed(1)+"%";$("rc").textContent=(100*d.result.confidence).toFixed(1)+"%";$("sq").textContent=d.squeeze_score.toFixed(5);$("top5").innerHTML=d.top5.map((r,i)=>`<div class="toprow"><span>0${i+1}</span><div>${esc(r.label)}<div class="barbg"><div class="bar" style="width:${Math.min(100,r.confidence*100)}%"></div></div></div><span class="score">${(100*r.confidence).toFixed(1)}%</span></div>`).join("")}catch(e){toast(e.message)}finally{$("run").disabled=false;$("run").textContent="Run robustness test ↗"}};
async function status(){try{let d=await req("/api/status");$("device").textContent=d.device;let vals=Object.values(d.models).flatMap(x=>Object.values(x));$("status").textContent=vals.filter(Boolean).length+"/4 models loaded";if(vals.every(Boolean))$("statusdot").classList.add("ok")}catch(e){$("status").textContent="Backend unavailable"}}
async function bench(){try{let d=await req("/api/benchmark");if(!d.available){$("bench").innerHTML=`<div class="panel empty"><h3>No benchmark data loaded</h3><p>${esc(d.message)}</p><button id="refresh">Check for results</button></div>`;$("refresh").onclick=bench;return}let rows=d.results.results;let cols=["module","model","clean_accuracy","fgsm_accuracy","pgd_accuracy","fgsm_attack_success_rate","pgd_attack_success_rate"];$("bench").innerHTML=`<div class="panel" style="overflow:auto"><table class="bench-table"><thead><tr>${cols.map(c=>`<th>${c.replaceAll("_"," ").toUpperCase()}</th>`).join("")}</tr></thead><tbody>${rows.map(r=>`<tr>${cols.map(c=>`<td>${["clean_accuracy","fgsm_accuracy","pgd_accuracy","fgsm_attack_success_rate","pgd_attack_success_rate"].includes(c)?(r[c]*100).toFixed(2)+"%":esc(r[c])}</td>`).join("")}</tr>`).join("")}</tbody></table></div><div class="panel chartpanel"><canvas id="chart"></canvas></div><div class="notice">${esc(d.results.warning||"Measured benchmark results.")}</div>`;if(chart)chart.destroy();chart=new Chart($("chart"),{type:"bar",data:{labels:rows.map(r=>r.module.toUpperCase()+" / "+r.model),datasets:[{label:"Clean",data:rows.map(r=>r.clean_accuracy),backgroundColor:"#48e0d0"},{label:"FGSM",data:rows.map(r=>r.fgsm_accuracy),backgroundColor:"#5d8dff"},{label:"PGD",data:rows.map(r=>r.pgd_accuracy),backgroundColor:"#a28bff"}]},options:{responsive:true,scales:{y:{min:0,max:1,ticks:{color:"#91a1b8"},grid:{color:"#223249"}},x:{ticks:{color:"#91a1b8"}}},plugins:{legend:{labels:{color:"#c8d5e5"}}}}})}catch(e){toast(e.message)}}
$("refresh").onclick=bench;status();


async function getExperimentForm() {
  if (!file) throw Error("Choose an image first.");
  const attack = document.querySelector('input[name="attack"]:checked').value;
  const f = new FormData();
  f.append("image", file); f.append("task", task); f.append("attack", attack);
  f.append("epsilon", $("eps").value); f.append("steps", $("steps").value);
  return { f, attack };
}
$("compare").onclick = async () => {
  try {
    const {f} = await getExperimentForm();
    $("compare").disabled = true; $("compare").textContent = "Comparing models…";
    const d = await req("/api/compare", {method:"POST", body:f});
    const rows = Object.entries(d.models).map(([name, r]) =>
      `<tr><th>${esc(name)}</th><td>${esc(r.clean_label)} (${(r.clean_confidence*100).toFixed(1)}%)</td><td>${esc(r.result_label)} (${(r.result_confidence*100).toFixed(1)}%)</td><td>${r.prediction_changed ? "Yes" : "No"}</td></tr>`).join("");
    $("comparison").innerHTML = `<b>Same image and settings</b><table><thead><tr><th>Model</th><th>Clean prediction</th><th>After attack</th><th>Changed?</th></tr></thead><tbody>${rows}</tbody></table><p class="fine">${esc(d.note)}</p>`;
    $("comparison").classList.remove("hidden");
  } catch(e) { toast(e.message); }
  finally { $("compare").disabled = false; $("compare").textContent = "Compare baseline vs defended"; }
};
$("report").onclick = async () => {
  try {
    const {f} = await getExperimentForm();
    f.append("variant", $("variant").value);
    $("report").disabled = true; $("report").textContent = "Creating report…";
    const r = await fetch("/api/report", {method:"POST", body:f});
    if (!r.ok) { const d = await r.json(); throw Error(d.detail || "Report generation failed"); }
    const blob = await r.blob(), url = URL.createObjectURL(blob), a = document.createElement("a");
    a.href = url; a.download = "aishield_security_report.pdf"; document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
  } catch(e) { toast(e.message); }
  finally { $("report").disabled = false; $("report").textContent = "Download PDF report"; }
};
async function loadSecurityScore() {
  const el = $("securityscore"); if (!el) return;
  try {
    const d = await req("/api/security-score");
    if (!d.available) { el.innerHTML = `<div><small>EXPERIMENTAL AI SECURITY SCORE</small><h3>Not available yet</h3><p>${esc(d.message)}</p></div>`; return; }
    const comps = Object.entries(d.components).map(([k,v]) => `${esc(k.replaceAll("_"," "))}: ${(v*100).toFixed(1)}%`).join(" · ");
    el.innerHTML = `<div><small>EXPERIMENTAL AI SECURITY SCORE</small><h3>Measured robustness summary</h3><p>${comps}</p><p class="score-warning">${esc(d.formula)}<br>${esc(d.warning)}</p></div><div class="score-number">${d.score}<small style="font-size:14px">/100</small></div>`;
  } catch(e) { el.innerHTML = `<div><small>EXPERIMENTAL AI SECURITY SCORE</small><p>${esc(e.message)}</p></div>`; }
}
const oldBench = bench;
bench = async function() { await oldBench(); await loadSecurityScore(); };
