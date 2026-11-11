import {api,closeDrawer,el,icon,navigationParams,prefs,savePref} from "./core.js";
import {disposeCharts} from "./charts.js";
import {captureViewport,restoreViewport} from "./viewport.js";
import {renderOverview} from "./views/overview.js";
import {renderLearner} from "./views/learner.js";
import {renderWorkers} from "./views/workers.js";
import {renderGroups} from "./views/groups.js";
import {renderBepa} from "./views/bepa.js";
import {renderEvaluation} from "./views/evaluation.js";
import {renderRecovery} from "./views/recovery.js";
import {renderEvents} from "./views/events.js";

const PAGES=[
  {id:"overview",label:"总览",title:"总览",subtitle:"训练、采样与运行证据",eyebrow:"OPERATIONS OVERVIEW",group:"运行"},
  {id:"learner",label:"Learner",title:"Learner",subtitle:"训练阶段、前向计量与有效 optimizer 提交",eyebrow:"LEARNER TELEMETRY",group:"运行"},
  {id:"workers",label:"Workers & cluster",title:"Workers & cluster",subtitle:"节点、请求槽位、KV 与物理资源",eyebrow:"CLUSTER CAPACITY",group:"运行"},
  {id:"groups",label:"Groups & trajectories",title:"Groups & trajectories",subtitle:"原生题组、训练投影与轨迹文件",eyebrow:"DATA LINEAGE",group:"运行"},
  {id:"bepa",label:"BEPA",title:"BEPA",subtitle:"Teacher budget、plan、guided rerun 与 cache",eyebrow:"TEACHER-ASSISTED EVIDENCE",group:"证据"},
  {id:"evaluation",label:"Evaluation",title:"Evaluation",subtitle:"Frozen taskset 的 baseline 与 native checkpoint 结果",eyebrow:"INDEPENDENT ARC EVALUATION",group:"证据"},
  {id:"recovery",label:"Recovery & checkpoints",title:"Recovery & checkpoints",subtitle:"持久训练状态、恢复 lineage 与 checkpoint inventory",eyebrow:"DURABLE TRAINING STATE",group:"证据"},
  {id:"events",label:"Events & logs",title:"Events & logs",subtitle:"运行事件、来源引用与角色日志",eyebrow:"OBSERVABILITY",group:"系统"},
];
const VIEW={overview:renderOverview,learner:renderLearner,workers:renderWorkers,groups:renderGroups,bepa:renderBepa,evaluation:renderEvaluation,recovery:renderRecovery,events:renderEvents};
const state={workspace:null,runId:null,sourceId:null,defaultSource:null,localSourceId:"local",sources:[],sourceKey:null,route:"overview",theme:"light",auto:true,timer:null,loading:false,lastSignature:"",lastError:null,scroll:{}};
let requestSequence=0;
let renderedViewportKey=null;
const $=(id)=>document.getElementById(id);
function routeFromHash(){const value=(location.hash||"").replace(/^#\/?/,"").split("?")[0];return PAGES.some((page)=>page.id===value)?value:"overview";}
function currentPage(){return PAGES.find((page)=>page.id===state.route)||PAGES[0];}
function writeNavigationUrl({route=state.route,sourceId,runId,setSource=false,setRun=false,push=false}={}){
  const params=new URLSearchParams(location.search);
  if(setSource){if(sourceId)params.set("source",sourceId);else params.delete("source");}
  if(setRun){if(runId)params.set("run_id",runId);else params.delete("run_id");}
  const query=params.toString(),url=location.pathname+(query?"?"+query:"")+"#/"+route;
  history[push?"pushState":"replaceState"](history.state,"",url);
}
function workspaceSourceKey(workspace){const source=workspace&&workspace.source||{},identity=(workspace&&workspace.selection&&workspace.selection.source_identity)||source.identity||{};return [state.sourceId||state.localSourceId,source.kind||"unknown",source.deployment_id||"",source.deployment||"",source.root||"",identity.container_id||"",identity.profile_sha256||"",identity.instance_run_id||""].join("|");}
function sourceParams(params={}){return {...params,source:state.sourceId||state.localSourceId};}
function renderSources(){const select=$("source-select");select.replaceChildren(...state.sources.map((item)=>el("option",{value:item.id},item.label)));select.value=state.sourceId||"";select.disabled=state.sources.length<2;}
function setTheme(theme){
  state.theme=theme==="dark"?"dark":"light";
  document.documentElement.dataset.theme=state.theme;
  document.documentElement.style.colorScheme=state.theme;
  $("theme-toggle").replaceChildren(icon(state.theme==="dark"?"sun":"moon"));
  $("theme-toggle").title=state.theme==="dark"?"切换至浅色主题":"切换至深色主题";
  savePref("theme",state.theme);
}
function navLink(page){
  const active=state.route===page.id;
  return el("a",{href:"#/"+page.id,className:"nav-link "+(active?"active":""),"aria-current":active?"page":null,title:page.label},
    el("span",{className:"nav-icon"},icon(page.id,18)),el("span",{className:"nav-label"},page.label),active?el("span",{className:"nav-active-mark"}):null);
}
function renderNav(){
  const nav=$("nav");const groups=["运行","证据","系统"];
  nav.replaceChildren(...groups.map((group)=>{
    const pages=PAGES.filter((page)=>page.group===group);
    return el("div",{className:"nav-section"},el("div",{className:"nav-section-label"},group),...pages.map(navLink));
  }));
}
function updateHeader(){
  const workspace=state.workspace,page=currentPage();
  $("page-title").textContent=page.title;
  $("page-description").textContent=page.subtitle;
  $("page-eyebrow").textContent=page.eyebrow;
  $("breadcrumb-current").textContent=page.title;
  $("footer-updated").textContent=workspace&&workspace.generated_at?new Date(typeof workspace.generated_at==="number"?workspace.generated_at*1000:workspace.generated_at).toLocaleTimeString("zh-CN",{hour12:false}):"—";
  $("generated-at").textContent=workspace&&workspace.generated_at?"采集 "+new Date(typeof workspace.generated_at==="number"?workspace.generated_at*1000:workspace.generated_at).toLocaleTimeString("zh-CN",{hour12:false}):"等待快照";
  $("build-tag").textContent="build "+(workspace&&workspace.build||"—");
  $("footer-source").textContent=workspace&&workspace.source&&workspace.source.kind?workspace.source.kind+" · watch JSON · run records · BEPA ledger":"watch JSON · run records · BEPA ledger";
  const run=workspace&&workspace.run||{};
  const observationOnly=!!(workspace&&workspace.selection&&workspace.selection.observation_only===true&&!workspace.selection.run_id);
  $("sidebar-run").textContent=observationOnly?"当前 Learner 容器":(run.run_id||"—");
  const watch=workspace&&workspace.watch||{};
  const liveSelected=workspace&&workspace.selection&&workspace.selection.live_selected===true;
  const status=$("sync-state"),dot=$("sidebar-source-dot");
  let label="实时状态未知",tone="warn",detail="没有可用快照";
  if(workspace&&workspace.selection&&workspace.selection.view_mode==="history"){label="历史复盘";tone="warn";detail="实时指标已隔离";}
  else if(workspace&&workspace.selection&&workspace.selection.view_mode==="unavailable"){label="即时观测不可用";tone="warn";detail=watch.error||"等待新鲜状态";}
  else if(watch.ok===true){
    const age=watch.age_seconds;
    const lifecycle=watch.snapshot&&watch.snapshot.lifecycle&&watch.snapshot.lifecycle.status;
    const stale=typeof age==="number"&&age>60;
    label=stale?"快照过期":(lifecycle==="running"?"实时快照":"最近快照");
    tone=stale?"warn":(lifecycle==="running"?"good":"neutral");
    detail=typeof age==="number"?"快照 "+Math.round(age)+" 秒前":(lifecycle||"watch 已连接");
  }else if(state.lastError&&state.sourceId!==state.localSourceId){label="远端 source 不可用";tone="bad";detail=state.lastError;}
  else if(watch.error){label="watch 不可用";tone="bad";detail=watch.error;}
  status.className="sync-pill sync-"+tone;
  status.replaceChildren(el("i",{}),el("span",{},label));
  dot.className="source-dot source-"+tone;
  $("sidebar-source-title").textContent=label;
  $("sidebar-source-detail").textContent=detail;
  const select=$("run-select");
  const runs=Array.isArray(workspace&&workspace.runs)?workspace.runs:[];
  const selection=workspace&&workspace.selection||{};
  const selected=selection.run_id||run.run_id||"";
  const options=[];
  if(observationOnly)options.push(el("option",{value:"",selected:!state.runId},"当前 Learner 容器"));
  else if(selection.active_run_id||selected){
    const label=selection.view_mode==="history"?"默认历史 run · "+String(selected).slice(0,28):selection.active_run_id?"当前活动 run · "+String(selection.active_run_id).slice(0,28):"默认 run · "+String(selected).slice(0,28);
    options.push(el("option",{value:"",selected:!state.runId},label));
  }else if(!runs.length)options.push(el("option",{value:"",selected:true},"暂无可用 run"));
  select.replaceChildren(...options,...runs.map((item)=>{
    const date=item.created_at?new Date(typeof item.created_at==="number"?item.created_at*1000:item.created_at).toLocaleDateString("zh-CN"):"日期未知";
    const active=item.active===true||item.run_id===selection.active_run_id;
    const option=el("option",{value:item.run_id},date+" · "+String(item.run_id||"").slice(0,28)+(active?" · active":"")+" · "+(item.status||"unknown"));
    return option;
  }));
  select.value=state.runId||"";
  if(!select.options.length){select.append(el("option",{value:""},"暂无可用 run"));select.disabled=true;}else select.disabled=false;
  renderNav();
}
function notice(){
  const workspace=state.workspace;
  const root=$("global-notice");
  root.replaceChildren();
  if(state.lastError){root.append(el("div",{className:"notice notice-error"},el("span",{className:"notice-mark"},"!"),el("div",{},el("b",{},state.sourceId!==state.localSourceId?"远端数据源不可用":"读取失败"),el("p",{},state.lastError))));return;}
  if(workspace&&workspace.selection&&workspace.selection.view_mode==="history"){
    root.append(el("div",{className:"notice notice-info"},el("span",{className:"notice-mark"},"i"),el("div",{},el("b",{},"历史 run 模式"),el("p",{},"此页面只使用所选 run 的记录；活动进程的 live 状态不会混入历史统计。"))));
  }
}
function signature(workspace){
  if(!workspace)return "";
  const copy=JSON.parse(JSON.stringify(workspace));
  delete copy.generated_at;
  if(copy.watch)delete copy.watch.age_seconds;
  return (state.sourceId||state.localSourceId)+"|"+JSON.stringify(copy);
}
function viewportKey(){return (state.sourceId||state.localSourceId)+"|"+state.route;}
function captureScroll(){
  state.scroll[viewportKey()]=captureViewport();
  savePref("scroll",state.scroll);
}
function restoreScroll(){restoreViewport(state.scroll[viewportKey()]||{});}
function renderWorkspace(force=false){
  if(!state.workspace)return;
  const nextSignature=signature(state.workspace);
  // The reader may have scrolled while the API request was pending. Capture
  // at DOM commit, before header/chart changes, never at request dispatch.
  if(renderedViewportKey===viewportKey())captureScroll();
  updateHeader();notice();
  if(!force&&nextSignature===state.lastSignature)return;
  disposeCharts();
  const target=$("view-slot");
  const renderer=VIEW[state.route]||renderOverview;
  const view=renderer(state.workspace,{sourceId:state.sourceId});
  const element=view&&view.element?view.element:view;
  target.replaceChildren(element);
  state.lastSignature=nextSignature;
  restoreScroll();
  renderedViewportKey=viewportKey();
  if(view&&typeof view.mount==="function")requestAnimationFrame(()=>view.mount());
}
async function refresh(force=false){
  if(!state.sourceId)return;
  if(state.loading&&!force)return;
  const token=++requestSequence;
  state.loading=true;
  $("refresh-button").disabled=true;
  $("refresh-button").classList.add("is-loading");
  $("refresh-icon").replaceChildren(icon("refresh",15));
  try{
    const params={};if(state.runId)params.run_id=state.runId;
    let workspace=await api("/api/workspace",sourceParams(params));
    if(token!==requestSequence)return;
    let key=workspaceSourceKey(workspace);
    if(state.sourceKey===null){
      state.sourceKey=key;
    }else if(key!==state.sourceKey){state.sourceKey=key;}
    state.workspace=workspace;state.lastError=null;
    renderWorkspace(force);
  }catch(error){
    if(token!==requestSequence)return;
    state.lastError=error.message||String(error);
    if(state.sourceId!==state.localSourceId){state.workspace=null;state.sourceKey=null;state.lastSignature="";disposeCharts();updateHeader();notice();$("view-slot").replaceChildren(el("div",{className:"error-state"},icon("alert",22),el("h2",{},"远端数据源暂不可用"),el("p",{},state.lastError),el("button",{className:"button button-primary",onClick:()=>refresh(true)},"重试")));}
    else if(!state.workspace){$("view-slot").replaceChildren(el("div",{className:"error-state"},icon("alert",22),el("h2",{},"无法读取 dashboard workspace"),el("p",{},state.lastError),el("button",{className:"button button-primary",onClick:()=>refresh(true)},"重试")));}
    else notice();
  }finally{
    if(token===requestSequence){state.loading=false;$("refresh-button").disabled=false;$("refresh-button").classList.remove("is-loading");}
  }
}
function selectSource(sourceId){
  if(!state.sources.some((item)=>item.id===sourceId)||sourceId===state.sourceId)return;
  requestSequence++;
  state.loading=false;state.sourceId=sourceId;state.runId=null;state.sourceKey=null;state.workspace=null;state.lastError=null;state.lastSignature="";
  writeNavigationUrl({sourceId,runId:null,setSource:true,setRun:true});
  renderSources();disposeCharts();updateHeader();notice();
  $("view-slot").replaceChildren(el("div",{className:"loading-state"},el("span",{className:"loader"}),el("b",{},"正在读取所选数据源"),el("small",{},"运行与历史请求已重置")));
  refresh(true);
}
function setRoute(route){
  if(!PAGES.some((page)=>page.id===route))route="overview";
  if(route===state.route){renderWorkspace(true);return;}
  captureScroll();state.route=route;savePref("route",route);writeNavigationUrl({route,push:true});
  renderWorkspace(true);$("sidebar").classList.remove("open");$("mobile-backdrop").hidden=true;
}
function syncRouteFromLocation(){
  const route=routeFromHash(),params=navigationParams();
  const sourceId=params.has("source")?params.get("source"):state.defaultSource;
  const runId=params.has("run_id")?params.get("run_id"):null;
  if(!state.sources.some((item)=>item.id===sourceId)){
    requestSequence++;state.sourceId=null;state.workspace=null;state.runId=null;
    state.lastError="URL source 不可用："+sourceId;updateHeader();notice();
    $("view-slot").replaceChildren(el("div",{className:"error-state"},state.lastError));return;
  }
  const changed=sourceId!==state.sourceId||runId!==state.runId;
  if(route!==state.route)captureScroll();
  state.route=route;savePref("route",route);
  if(changed){
    requestSequence++;state.sourceId=sourceId;state.runId=runId;state.sourceKey=null;
    state.workspace=null;state.lastSignature="";state.lastError=null;renderSources();refresh(true);
  }else renderWorkspace(true);
}
function showToast(message){
  const region=$("toast-region"),item=el("div",{className:"toast"},message);
  region.append(item);setTimeout(()=>item.remove(),3200);
}
function ensureCharts(){
  if(window.echarts)return Promise.resolve(true);
  const existing=document.getElementById("echarts-vendor");
  if(!existing)return Promise.resolve(false);
  return new Promise((resolve)=>{let settled=false;const finish=(value)=>{if(settled)return;settled=true;resolve(value);};existing.addEventListener("load",()=>finish(Boolean(window.echarts)),{once:true});existing.addEventListener("error",()=>finish(false),{once:true});setTimeout(()=>finish(Boolean(window.echarts)),5000);});
}
function boot(){
  const saved=prefs();
  state.route=location.hash?routeFromHash():(saved.route||"overview");
  const requested=navigationParams();
  state.runId=requested.has("run_id")?requested.get("run_id"):null;
  state.sourceId=null;
  state.theme=saved.theme||"light";
  state.auto=saved.auto===undefined?true:Boolean(saved.auto);
  state.scroll=saved.scroll||{};
  selectedRouteHash();
  setTheme(state.theme);
  $("auto-refresh").checked=state.auto;
  $("theme-toggle").addEventListener("click",()=>{captureScroll();setTheme(state.theme==="dark"?"light":"dark");renderWorkspace(true);});
  $("refresh-button").addEventListener("click",()=>refresh(true));
  $("auto-refresh").addEventListener("change",()=>{state.auto=$("auto-refresh").checked;savePref("auto",state.auto);});
  $("run-select").addEventListener("change",()=>{state.runId=$("run-select").value||null;writeNavigationUrl({sourceId:state.sourceId,runId:state.runId,setSource:true,setRun:true});refresh(true);});
  $("source-select").addEventListener("change",()=>selectSource($("source-select").value));
  $("mobile-menu").replaceChildren(icon("menu",18));
  $("mobile-menu").addEventListener("click",()=>{$("sidebar").classList.toggle("open");$("mobile-backdrop").hidden=!$("sidebar").classList.contains("open");});
  $("mobile-backdrop").addEventListener("click",()=>{$("sidebar").classList.remove("open");$("mobile-backdrop").hidden=true;});
  $("drawer-backdrop").addEventListener("click",closeDrawer);
  document.addEventListener("keydown",(event)=>{if(event.key==="Escape")closeDrawer();});
  document.addEventListener("click",(event)=>{const anchor=event.target.closest("a[href^='#/']");if(anchor){const route=anchor.getAttribute("href").slice(2);if(PAGES.some((page)=>page.id===route)){event.preventDefault();setRoute(route);}}});
  window.addEventListener("hashchange",syncRouteFromLocation);
  window.addEventListener("popstate",syncRouteFromLocation);
  document.addEventListener("visibilitychange",()=>{if(!document.hidden&&state.auto)refresh(false);});
  api("/api/sources").then((catalog)=>{
    state.sources=Array.isArray(catalog.sources)?catalog.sources:[];
    state.defaultSource=catalog.default_source;
    state.localSourceId=catalog.local_source_id||"local";
    const requested=navigationParams();
    if(requested.has("source")){
      const requestedSource=requested.get("source");
      if(!state.sources.some((item)=>item.id===requestedSource))throw new Error("URL source 不可用："+requestedSource);
      state.sourceId=requestedSource;
    }else state.sourceId=catalog.default_source;
    if(!state.sources.some((item)=>item.id===state.sourceId))throw new Error("服务器没有可用的数据来源");
    renderSources();
    return ensureCharts();
  }).then(()=>refresh(true)).catch((error)=>{
    state.lastError=error.message||String(error);updateHeader();notice();
    $("view-slot").replaceChildren(el("div",{className:"error-state"},icon("alert",22),el("h2",{},"无法读取数据来源"),el("p",{},state.lastError),el("button",{className:"button button-primary",onClick:()=>location.reload()},"重试")));
  });
  state.timer=setInterval(()=>{if(state.auto&&!document.hidden)refresh(false);},2000);
}
function selectedRouteHash(){const route=routeFromHash();if(location.hash)state.route=route;}
if(document.readyState==="loading")document.addEventListener("DOMContentLoaded",boot);else boot();
