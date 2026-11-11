import {DASH,ago,api,badge,dataTable,el,emptyState,fmtBytes,fmtDate,fmtInt,grid,jsonDetails,kvRows,makeLink,panel,prefs,safeArray,savePref,sourceNote} from "../core.js";
import {eventDrawer} from "./common.js";

let selectedSeverity="all",selectedRole="learner",tail=200,logSearch="";
const LOG_ROLE_PATTERN=/^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$/;
const SEVERITY_FILTERS=["all","high","warning","info","unknown","unrecorded"];
export function logRoleCatalog(run){
  const roles=new Set(safeArray(run&&run.logs_available).filter((role)=>typeof role==="string"&&LOG_ROLE_PATTERN.test(role)));
  const manifest=run&&run.manifest||{};
  for(const process of safeArray(run&&run.manifest&&run.manifest.processes)){
    if(!process||typeof process!=="object"||typeof process.log!=="string"||!process.log.trim())continue;
    const role=process.role,workerId=process.worker_id;
    let key="";
    if(role==="worker")key=workerId;
    else if(["manager","learner","teacher"].includes(role))key=role;
    else if(typeof role==="string")key=role+(typeof workerId==="string"&&workerId?"-"+workerId:"");
    if(typeof key==="string"&&LOG_ROLE_PATTERN.test(key))roles.add(key);
  }
  const declaredWorkers=new Set();
  const addWorkers=(workers)=>{
    if(Array.isArray(workers))for(const worker of workers){
      const key=typeof worker==="string"?worker:worker&& (worker.worker_id||worker.id);
      if(typeof key==="string"&&LOG_ROLE_PATTERN.test(key))declaredWorkers.add(key);
    }
    else if(workers&&typeof workers==="object")for(const key of Object.keys(workers))if(LOG_ROLE_PATTERN.test(key))declaredWorkers.add(key);
  };
  addWorkers(manifest.endpoints&&manifest.endpoints.workers);
  addWorkers(manifest.topology&&manifest.topology.workers);
  for(const worker of safeArray(manifest.membership&&manifest.membership.active_workers)){
    if(typeof worker.worker_id==="string"&&LOG_ROLE_PATTERN.test(worker.worker_id))declaredWorkers.add(worker.worker_id);
  }
  const sort=(values)=>[...values].sort((a,b)=>a.localeCompare(b,"en",{numeric:true}));
  return {available:sort(roles),unavailable:sort([...declaredWorkers].filter((role)=>!roles.has(role)))};
}
export function preferredLogRole(roles,preferred,current){
  if(roles.includes(preferred))return preferred;
  if(roles.includes(current))return current;
  return roles[0]||"";
}
export function severityLabel(item){
  const value=item&&item.severity;
  if(value===null||value===undefined||String(value).trim()==="")return "未记录";
  return String(value).toLowerCase()==="unknown"?"未知":String(value);
}
export function policyLabel(item){
  const value=item&&item.policy_version;
  if(typeof value==="number"&&Number.isInteger(value)&&value>=0)return "v"+String(value);
  return item&&item.policy_applicability==="not_applicable"?"不适用":"未记录";
}
function buildEvents(workspace){
  const combined=safeArray(workspace.events);
  const seen=new Set();
  return combined.filter((item)=>{const id=item.id||[item.at,item.type,item.source,item.title].join("|");if(seen.has(id))return false;seen.add(id);return true;});
}
function severityFilter(item){
  if(selectedSeverity==="all")return true;
  const severity=String(item.severity||"").trim().toLowerCase();
  if(selectedSeverity==="unrecorded")return !severity;
  if(selectedSeverity==="unknown")return severity==="unknown";
  if(selectedSeverity==="high")return ["critical","error","failed","high"].includes(severity);
  if(selectedSeverity==="warning")return ["warning","warn","medium"].includes(severity);
  return ["info","informational"].includes(severity);
}
function eventTable(workspace,items){
  return dataTable("event-feed",items,[
    {key:"at",label:"时间",sortValue:(row)=>row.at??row.completed_at??row.created_at,render:(row)=>fmtDate(row.at??row.completed_at??row.created_at)},
    {key:"severity",label:"级别",render:(row)=>badge(severityLabel(row))},
    {key:"type",label:"Event type"},
    {key:"title",label:"摘要",render:(row)=>el("span",{className:"event-title-cell"},row.title||row.message||"事件")},
    {key:"source",label:"来源"},
    {key:"policy_version",label:"Policy",numeric:true,render:(row)=>el("span",{title:[row.policy_role,row.policy_basis].filter(Boolean).join(" · ")},policyLabel(row))},
    {key:"ref",label:"Evidence",render:(row)=>row.ref?makeLink("查看",()=>eventDrawer(row),"text-button"):DASH},
  ],{pageSize:14,note:"事件来自 run / runtime / BEPA 证据；打开行查看 detail 与 references",filter:severityFilter,onRow:eventDrawer});
}
function logControls(workspace,sourceId,catalog){
  const roles=catalog.available;
  const options=[...roles.map((role)=>el("option",{value:role,selected:role===selectedRole},role)),
    ...catalog.unavailable.map((role)=>el("option",{value:role,disabled:true},role+"（API 未提供日志）"))];
  if(!options.length)options.push(el("option",{value:""},"无可用日志角色"));
  const roleSelect=el("select",{"aria-label":"日志角色",className:"control-select",disabled:!roles.length},options);
  roleSelect.addEventListener("change",()=>{selectedRole=roleSelect.value;savePref("logRole",selectedRole);loadLogs(workspace,sourceId,catalog);});
  const tailSelect=el("select",{"aria-label":"尾部行数",className:"control-select"},[100,200,500,1000].map((size)=>el("option",{value:size,selected:Number(size)===Number(tail)},size+" lines")));
  tailSelect.addEventListener("change",()=>{tail=Number(tailSelect.value);savePref("logTail",tail);loadLogs(workspace,sourceId,catalog);});
  const search=el("label",{className:"table-search log-search"},el("input",{type:"search",placeholder:"筛选日志文本",value:logSearch,"aria-label":"筛选日志文本"}));
  search.querySelector("input").addEventListener("input",()=>{logSearch=search.querySelector("input").value;savePref("logSearch",logSearch);renderLogLines(lastLogs,logIdentity(workspace,sourceId,selectedRole));});
  return el("div",{className:"log-controls"},roleSelect,tailSelect,search);
}
let logPane,logMeta,lastLogs={lines:[]},logRequestSequence=0,activeWorkspace=null;
function selectedRunId(workspace){return workspace&&workspace.selection&&workspace.selection.run_id||workspace&&workspace.run&&workspace.run.run_id||"";}
function logIdentity(workspace,sourceId,role){
  const selection=workspace&&workspace.selection||{},runId=selectedRunId(workspace);
  const runLabel=selection.view_mode==="history"?"历史 run":"所选 run";
  return {sourceId,runId,role,runLabel,text:`${sourceId||"source 未记录"} · ${runLabel} ${runId||"未记录"} · role ${role||"无可用角色"}`};
}
async function loadLogs(workspace,sourceId,catalog){
  if(!logPane)return;
  const roles=catalog.available;
  const role=selectedRole,identity=logIdentity(workspace,sourceId,role),runId=identity.runId;
  const requestId=++logRequestSequence,pane=logPane,meta=logMeta,params=new URLSearchParams(location.search);
  const routeSource=params.get("source"),routeRun=params.get("run_id");
  const isCurrent=()=>{
    const currentParams=new URLSearchParams(location.search),currentSource=currentParams.get("source"),currentRun=currentParams.get("run_id");
    return requestId===logRequestSequence&&pane===logPane&&meta===logMeta&&activeWorkspace===workspace&&pane.isConnected&&
      currentSource===routeSource&&currentRun===routeRun&&(!currentSource||currentSource===sourceId)&&(!currentRun||currentRun===runId);
  };
  if(!runId){lastLogs={lines:[],error:"没有 selected run；未请求角色日志。"};renderLogLines(lastLogs,identity);return;}
  if(!roles.length||!role){
    const error=catalog.unavailable.length?`manifest 声明了 ${catalog.unavailable.join("、")}，但 /api/logs 未提供这些角色的文本日志。`:"所选 run 没有可读取的角色日志。";
    lastLogs={lines:[],error};renderLogLines(lastLogs,identity);return;
  }
  lastLogs={lines:[]};renderLogLines({loading:true},identity);
  try{
    const data=await api("/api/logs",{source:sourceId,run_id:runId,role,tail});
    if(!isCurrent())return;
    if(data.error==="unknown role")data.error=`/api/logs 未提供 ${role} 角色的文本日志。`;
    lastLogs=data;renderLogLines(data,identity);
  }catch(error){if(!isCurrent())return;lastLogs={lines:[],error:error.message||String(error)};renderLogLines(lastLogs,identity);}
}
function renderLogLines(data,identity={}){
  if(!logPane)return;
  logMeta.replaceChildren(el("span",{className:"mono"},identity.text||"日志来源与 run 未记录"),data.file?el("span",{}," · "+data.file):null,data.bytes!==undefined?el("span",{}," · "+fmtBytes(data.bytes)):null,data.mtime?el("span",{}," · "+ago(data.mtime)):null,data.error?el("span",{className:"log-error"}," · "+data.error):null);
  if(data.loading){logPane.replaceChildren(el("div",{className:"log-empty"},"读取日志…"));return;}
  const lines=safeArray(data.lines).filter((line)=>!logSearch||line.toLocaleLowerCase().includes(logSearch.toLocaleLowerCase()));
  logPane.replaceChildren(...(lines.length?lines.map((line)=>el("div",{className:/error|traceback|exception|fatal/i.test(line)?"log-line log-line-error":/warn/i.test(line)?"log-line log-line-warn":"log-line"},line)): [el("div",{className:"log-empty"},data.error||"没有匹配的日志行")]));
}
export function renderEvents(workspace,{sourceId:requestedSourceId}={}){
  activeWorkspace=workspace;
  const sourceId=requestedSourceId||workspace&&workspace.source_id||new URLSearchParams(location.search).get("source")||"local";
  const catalog=logRoleCatalog(workspace&&workspace.run),roles=catalog.available;
  const prefsNow=prefs();
  selectedRole=preferredLogRole(roles,prefsNow.logRole,selectedRole);if(selectedRole&&selectedRole!==prefsNow.logRole)savePref("logRole",selectedRole);
  tail=Number(prefsNow.logTail)||tail;logSearch=prefsNow.logSearch||logSearch;selectedSeverity=SEVERITY_FILTERS.includes(prefsNow.eventSeverity)?prefsNow.eventSeverity:"all";
  const items=buildEvents(workspace);
  const feed=el("div",{className:"event-feed"});
  function filterButtons(){return SEVERITY_FILTERS.map((key)=>el("button",{type:"button",className:"filter-pill "+(selectedSeverity===key?"active":""),onClick:()=>{selectedSeverity=key;savePref("eventSeverity",key);filter.replaceChildren(...filterButtons());tableMount.replaceChildren(eventTable(workspace,items));}},key==="all"?"全部":key==="high"?"错误":key==="warning"?"警告":key==="info"?"信息":key==="unknown"?"未知级别":"未记录级别"));}
  const filter=el("div",{className:"filter-pills"},filterButtons());
  const tableMount=el("div",{className:"event-table-mount"},eventTable(workspace,items));
  const roleSummary=roles.length?`${roles.length} 个可读日志角色`:catalog.unavailable.length?"manifest 中有角色，但 API 未提供日志":"无可用日志角色";
  logMeta=el("div",{className:"log-meta"});
  logPane=el("div",{className:"log-pane"});
  lastLogs={lines:[]};logRequestSequence++;
  const logs=panel("角色日志",el("div",{className:"log-view"},el("div",{className:"tag-line"},el("span",{},roleSummary),el("span",{className:"muted"},"读取所选 run 的日志尾部，不触发训练或教师调用")),logControls(workspace,sourceId,catalog),logMeta,logPane),{eyebrow:"REDACTED ROLE LOGS",description:"请求绑定当前 source 与 selected run；历史 run 会明确标出。角色来自所选 run 的 logs_available/manifest；仅 API 已提供的角色可选择。日志 API 已做凭证值脱敏。",flush:true});
  return {element:el("div",{className:"view-stack"},
    el("header",{className:"section-title"},el("div",{className:"eyebrow"},"OBSERVABILITY"),el("h2",{},"Events & logs"),el("p",{},"按 severity 检索事件，下钻 source/ref，再查看有界角色日志尾部。")),
    grid([metricTile("Events",fmtInt(items.length),"events"),metricTile("Selected run",workspace.run&&workspace.run.run_id||DASH,"run",workspace.selection&&workspace.selection.live_selected?"Live-selected run":"Run-isolated"),metricTile("Log role",selectedRole,"role",tail+" line tail"),metricTile("Current filter",selectedSeverity,"severity","Filters persist locally")],"four"),
    panel("事件流",el("div",{className:"event-view"},filter,tableMount),{eyebrow:"TIMELINE",description:"最近的有界事件摘要；BEPA 为共享跨 run 台账。缺失级别标为未记录，未识别的动作保留未知；Policy 区分已知版本、不适用与未记录。",flush:true}),
    logs),mount:()=>loadLogs(workspace,sourceId,catalog)};
}
function metricTile(label,value,unit,meta){return el("article",{className:"metric-card"},el("div",{className:"metric-label"},label),el("div",{className:"metric-value"},value,el("span",{className:"metric-unit"},unit)),el("div",{className:"metric-meta"},meta));}
