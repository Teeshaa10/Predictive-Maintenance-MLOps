const API = "http://127.0.0.1:8000";
const $ = id => document.getElementById(id);

async function checkAPI(){
  try{
    const r=await fetch(`${API}/health`);
    const d=await r.json();
    $("modelStatus").textContent=d.model_loaded?"ONLINE":"MODEL MISSING";
  }catch(e){$("modelStatus").textContent="API OFFLINE";}
}
$("predictionForm").addEventListener("submit",async e=>{
  e.preventDefault();
  const payload={
    "Type":$("type").value,
    "Air temperature [K]":Number($("air").value),
    "Process temperature [K]":Number($("process").value),
    "Rotational speed [rpm]":Number($("speed").value),
    "Torque [Nm]":Number($("torque").value),
    "Tool wear [min]":Number($("wear").value)
  };
  const box=$("result");box.classList.remove("hidden");box.textContent="Running model prediction...";
  try{
    const response=await fetch(`${API}/predict`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
    const d=await response.json();
    if(!response.ok) throw new Error(d.detail||"Prediction failed");
    $("riskCard").textContent=d.risk_status;
    $("lifeCard").textContent=`${d.estimated_tool_life_remaining_minutes} min`;
    box.innerHTML=`<strong>Prediction result</strong><br>
      Failure prediction: <b>${d.failure_prediction===1?"Failure likely":"No failure predicted"}</b><br>
      Failure probability: <b>${(d.failure_probability*100).toFixed(2)}%</b><br>
      Risk status: <b>${d.risk_status}</b><br>
      Estimated tool life remaining: <b>${d.estimated_tool_life_remaining_minutes} minutes</b><br>
      Model: <b>${d.classifier}</b><br><small>${d.note}</small>`;
  }catch(err){box.textContent=`Error: ${err.message}. Check that the API is running and the model has been trained.`;}
});
checkAPI();
