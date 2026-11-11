import {openDrawer,DASH,ago,api,badge,compact,dataTable,el,emptyState,fmtBytes,fmtDate,fmtDuration,fmtInt,grid,jsonDetails,kvRows,metric,navigationParams,panel,safeArray,sourceNote,makeLink} from "../core.js";
import {openArtifact,liveSnapshot} from "./common.js";

const STAGES=["native_failure","teacher","plan","guided","cache","reservation","commit"];
const STAGE_LABELS={native_failure:"Native failure",teacher:"Teacher",plan:"Plan",guided:"Guided rerun",cache:"Student cache",reservation:"Reservation",commit:"Optimizer commit"};
function artifactButton(ref,label,runId){const lower=label.toLowerCase();const kind=lower.includes("plan")||label==="计划"?"plan artifact":lower.includes("teacher")?"Teacher trace":lower.includes("guided")||lower.includes("projection")?"guided artifact":"artifact";return ref?makeLink(label,()=>openArtifact(ref,runId,kind)):DASH;}
function stagePill(stage,key=""){
  const state=stage&&stage.state;
  if(!stage||!state)return badge("unknown");
  if(key==="cache"&&(stage.consumability==="unknown"||stage.central_claim_receipt_recheck==="not_performed"||String(state).startsWith("metadata_candidate")))return badge("metadata candidate · admission unknown","warn");
  return badge(state==="none"?"none":state);
}
function timelineDrawer(item,runId){
  const stages=item.stages||{};
  const stageRows=STAGES.map((key)=>{const value=stages[key]||{};const detail=[value.reason,value.job_id,value.plan_id,value.cache_entry_id,value.reservation_id,value.update_attempt_id,value.episode_id].filter(Boolean).join(" · ");const artifact=value.teacher_trace_ref||value.artifact_ref||value.plan_artifact_ref||value.source_receipt_ref||value.raw_artifact_ref||value.projection_artifact_ref||value.receipt_artifact_ref;const usage=key==="guided"?(value.attempts===undefined?null:(value.attempts+" / "+(value.attempt_cap===undefined?"?":value.attempt_cap))):key==="reservation"?value.count??value.committed_uses_observed:key==="commit"?value.new_policy_version:key==="cache"?(value.row_kind||value.provenance):value.reuse;const at=value.at||value.latest_at||value.updated_at||value.created_at;return {key,label:STAGE_LABELS[key],state:value.state,at,detail,usage,artifact,artifact_label:value.teacher_trace_ref?"Teacher trace":value.source_receipt_ref?"Teacher receipt":key==="plan"?"Open plan":key==="cache"?"Open guided artifact":"Open artifact",consumability:value.consumability||value.central_claim_receipt_recheck,reuse:value.reuse};});
  openDrawer("任务 "+(item.task_key||"未知"),el("div",{className:"drawer-stack"},
    panel("当前等待",kvRows([["阶段",item.wait&&item.wait.stage],["等待年龄",item.wait&&item.wait.elapsed_seconds!==null&&item.wait.elapsed_seconds!==undefined?fmtDuration(item.wait.elapsed_seconds):null],["起点",item.wait&&item.wait.since?fmtDate(item.wait.since):null],["时间依据",item.wait&&item.wait.time_basis],["Scope",item.scope_id]]),{eyebrow:"WAIT STATE"}),
    dataTable("bepa-task-"+(item.task_key||"unknown"),stageRows,[
      {key:"label",label:"阶段"},
      {key:"state",label:"状态",render:(row)=>stagePill(stages[row.key],row.key)},
      {key:"at",label:"时间",render:(row)=>row.at?fmtDate(row.at):"未知"},
      {key:"usage",label:"Attempts / usage",render:(row)=>row.usage===null||row.usage===undefined?DASH:String(row.usage)},
      {key:"consumability",label:"Admission evidence",render:(row)=>row.key==="cache"?(row.consumability||"unknown"):DASH},
      {key:"artifact",label:"Artifact",sortable:false,render:(row)=>row.artifact?artifactButton(row.artifact,row.artifact_label,runId):DASH},
      {key:"detail",label:"关联证据"},
      {key:"reuse",label:"Plan 可复用",render:(row)=>row.reuse===true?"是":row.reuse===false?"否":row.reuse?String(row.reuse):DASH},
    ],{search:false,pageSize:10}),
    jsonDetails("完整 task timeline",item)));
}
function scopeCard(scope){
  const max=typeof scope.max_total_tokens==="number"?scope.max_total_tokens:null;
  const used=typeof scope.used_tokens==="number"?scope.used_tokens:null;
  const reserved=typeof scope.reserved_tokens==="number"?scope.reserved_tokens:null;
  const known=max!==null&&used!==null,unknown=scope.usage_unknown;
  const hasUnknown=unknown===true||(typeof unknown==="number"&&unknown>0);
  const unknownLabel=typeof unknown==="number"?fmtInt(unknown):(unknown===true?"yes":unknown===false?"no":"—");
  const pct=known&&max>0?Math.max(0,Math.min(100,100*used/max)):null;
  const remaining=typeof scope.remaining_tokens==="number"?scope.remaining_tokens:
    known?Math.max(0,max-used-(reserved||0)):null;
  return el("article",{className:"scope-card"},
    el("div",{className:"scope-head"},el("b",{className:"mono"},compact(scope.scope_id||scope.name,34)),badge(hasUnknown?"usage unknown":"scope",hasUnknown?"warn":"neutral")),
    el("div",{className:"scope-numbers"},el("strong",{},known?fmtInt(used):DASH),el("span",{}," / "),el("span",{},max!==null?fmtInt(max):DASH),el("small",{},"tokens used")),
    el("div",{className:"meter "+(pct===null?"meter-unknown":"")},pct===null?null:el("i",{style:{width:pct+"%"}})),
    el("div",{className:"scope-foot"},el("span",{},"remaining "+(remaining!==null?fmtInt(remaining):DASH)),el("span",{},"reserved "+(reserved!==null?fmtInt(reserved):DASH)),el("span",{},"unknown "+unknownLabel)),
    scope.created_at?el("small",{className:"muted"},ago(scope.created_at)):null);
}
function stageTokens(stages){
  return el("div",{className:"timeline-stages"},STAGES.map((key,index)=>{const stage=stages&&stages[key]||{},at=stage.at||stage.latest_at||stage.updated_at||stage.created_at;return el("div",{className:"timeline-stage "+statusToneClass(stage.state)},el("span",{className:"timeline-stage-dot"},String(index+1)),el("div",{},el("b",{},STAGE_LABELS[key]),stagePill(stage,key),at?el("small",{},ago(at)):null));}));
}
function statusToneClass(value){const text=String(value||"").toLowerCase();return ["failed","error","blocked","unknown","outcome_unknown"].some((x)=>text.includes(x))?"stage-warn":["ready","completed","committed","cache_ready","verified"].some((x)=>text.includes(x))?"stage-good":"";}
function timelineTable(items,runId){
  return items.length?dataTable("bepa-timeline",items,[
    {key:"task_key",label:"Task",render:(row)=>el("span",{className:"mono"},compact(row.task_key,28))},
    {key:"wait.stage",label:"等待阶段",sortValue:(row)=>row.wait&&row.wait.stage,render:(row)=>row.wait&&row.wait.stage?badge(row.wait.stage,"warn"):badge("无等待")},
    {key:"wait.elapsed_seconds",label:"等待时长",numeric:true,sortValue:(row)=>row.wait&&row.wait.elapsed_seconds,render:(row)=>row.wait&&row.wait.elapsed_seconds!==null&&row.wait.elapsed_seconds!==undefined?fmtDuration(row.wait.elapsed_seconds):"未知"},
    {key:"scope_id",label:"Budget scope",render:(row)=>compact(row.scope_id,24)},
    {key:"stages.commit.state",label:"Commit",sortValue:(row)=>row.stages&&row.stages.commit&&row.stages.commit.state,render:(row)=>badge(row.stages&&row.stages.commit&&row.stages.commit.state||"unknown")},
  ],{pageSize:10,note:"打开行查看 stage-by-stage 时间与等待依据",onRow:(row)=>timelineDrawer(row,runId)}):emptyState("没有 BEPA task timeline","timeline 由预算、教师与 cache 台账提供。");
}
function sectionNote(bepa,key,label){const basis=bepa[key+"_basis"];const truncated=bepa[key+"_truncated"];return [label||"persisted rows",basis||"basis unavailable",truncated===true?"truncated":truncated===false?"complete within cap":"truncation unknown"].join(" · ");}
function pipelineEntry(bepa,id){return safeArray(bepa&&bepa.pipeline).find((item)=>item.id===id)||safeArray(bepa&&bepa.pipeline).find((item)=>item.label===id)||null;}
function statusSummary(title,counts,note){
  const pairs=Object.entries(counts||{});
  return pairs.length?el("div",{className:"status-count-grid"},pairs.map(([label,count])=>el("div",{className:"status-count"},el("span",{},label),el("b",{},fmtInt(count))))):emptyState(title,note||"The selected source has no row-level records for this section.");
}
function cacheSummary(cache,timeline){
  if(!cache||typeof cache!=="object")return emptyState("Cache evidence unknown","The selected BEPA source has no cache summary.");
  const byKind=cache.by_kind||{};
  const rows=Object.entries(byKind).map(([kind,count])=>({kind,count}));
  const withinLimits=cache.metadata_candidate_within_recorded_limits_observed_scope_tasks
    ??cache.metadata_candidate_within_recorded_limits;
  return el("div",{className:"drawer-stack"},
    kvRows([["Metadata candidates",cache.metadata_candidates],["Within recorded limits (observed tasks)",withinLimits],["Current epoch",cache.current_epoch],["Current policy",cache.current_policy_version],["Reservation rows observed",cache.reservation_rows_observed],["Reservation rows truncated",cache.reservation_rows_truncated],["Provenance recheck",cache.provenance_recheck],["Source basis",cache.basis]]),
    rows.length?dataTable("bepa-cache-kinds",rows,[{key:"kind",label:"Row kind"},{key:"count",label:"Observed rows",numeric:true}],{search:false,pageSize:8}):emptyState("No cache kind rows","No cache metadata rows were returned."),
    el("div",{className:"evidence-note evidence-warn"},el("strong",{},"Metadata candidate · consumability unknown"),el("p",{},"Central claim receipt and source artifact recheck were not performed by this view.")));
}
function teacherLogView(workspace,sourceId){
  const selection=workspace.selection||{},runId=workspace.run&&workspace.run.run_id||selection.run_id||null;
  const historical=selection.view_mode==="history"||Boolean(runId&&selection.active_run_id&&runId!==selection.active_run_id);
  const current=Boolean(runId&&selection.active_run_id===runId&&!historical);
  const mode=historical?"历史 run":current?"当前活动 run"+(selection.live_selected?" · live":" · 即时观测不可用"):"所选 run";
  const source=sourceId||navigationParams().get("source")||"server default";
  const logMeta=el("div",{className:"log-meta"},"source "+source+" · "+mode+" · run "+(runId||"—"));
  const logPane=el("div",{className:"log-pane"},el("div",{className:"log-empty"},"等待 Teacher 日志"));
  const events=safeArray(workspace.bepa&&workspace.bepa.events).slice(0,8);
  const eventBody=events.length?dataTable("bepa-teacher-ledger-events",events,[
    {key:"at",label:"时间",render:(row)=>row.at?fmtDate(row.at):DASH},
    {key:"type",label:"类型"},
    {key:"title",label:"事件"},
    {key:"detail",label:"Reason"},
  ],{search:false,pageSize:8,note:"所选 source 的 BEPA event ledger · 可能跨 scope · 不是角色文本日志"})
    :emptyState("当前 workspace 没有 BEPA ledger 事件行","这表示事件行未提供，不代表计数为 0。");
  async function loadTeacherLog(){
    if(!logPane.isConnected)return;
    if(!runId){logPane.replaceChildren(el("div",{className:"log-empty"},"没有可读取的 selected run；未请求 Teacher 日志。"));return;}
    logPane.replaceChildren(el("div",{className:"log-empty"},"读取 Teacher 日志…"));
    try{
      const data=await api("/api/logs",{source:sourceId,run_id:runId,role:"teacher",tail:200});
      if(!logPane.isConnected)return;
      logMeta.replaceChildren(el("span",{},data.file||"teacher.log"),el("span",{}," · "),el("span",{},fmtBytes(data.bytes)),el("span",{}," · "),el("span",{},data.mtime?ago(data.mtime):"mtime 未知"),el("span",{}," · source "+source),el("span",{}," · "+mode),data.error?el("span",{className:"log-error"}," · 读取错误："+data.error):null);
      const lines=safeArray(data.lines);
      if(data.error){logPane.replaceChildren(el("div",{className:"log-empty log-error"},"Teacher 日志读取错误："+data.error));return;}
      if(!lines.length){
        const emptyText=data.bytes===0?"此角色尚无文字日志（0 B）；进度请看下方 BEPA ledger 事件。":"Teacher 日志尾部没有可显示行；进度请看下方 BEPA ledger 事件。";
        logPane.replaceChildren(el("div",{className:"log-empty"},emptyText));return;
      }
      logPane.replaceChildren(...lines.map((line)=>el("div",{className:/error|traceback|exception|fatal/i.test(line)?"log-line log-line-error":/warn/i.test(line)?"log-line log-line-warn":"log-line"},line)));
    }catch(error){
      if(!logPane.isConnected)return;
      logMeta.replaceChildren(el("span",{},"source "+source),el("span",{}," · "+mode),el("span",{}," · run "+runId),el("span",{className:"log-error"}," · 读取错误："+(error.message||String(error))));
      logPane.replaceChildren(el("div",{className:"log-empty log-error"},"Teacher 日志读取错误："+(error.message||String(error))));
    }
  }
  return {element:el("div",{className:"drawer-stack"},
    panel(historical?"历史 Teacher 日志":"Teacher 日志",el("div",{className:"log-view"},logMeta,logPane),{eyebrow:historical?"HISTORICAL RUN LOG":"SELECTED SOURCE / RUN · ROLE TEACHER",description:"日志按 source、run_id 与 teacher 角色读取；历史 run 会单独标记。",flush:true}),
    panel("近期 BEPA ledger 事件",eventBody,{eyebrow:"EVENT LEDGER",description:"事件来自 source 的 BEPA 台账；它们不等同于 Teacher 角色文本日志。",flush:true})),mount:loadTeacherLog};
}
export function renderBepa(workspace,context={}){
  const bepa=workspace.bepa||{},runId=workspace.run&&workspace.run.run_id;
  const teacherLogs=teacherLogView(workspace,context.sourceId);
  const current=liveSnapshot(workspace);
  const snapshot=workspace.watch&&workspace.watch.snapshot||{};
  const freshness=snapshot.freshness&&snapshot.freshness.bepa||null;
  const watchAge=workspace.watch&&typeof workspace.watch.age_seconds==="number"?workspace.watch.age_seconds:null;
  const freshnessAge=freshness&&typeof freshness.age_seconds==="number"?freshness.age_seconds+(watchAge||0):watchAge;
  const teacherFresh=Boolean(current&&freshness&&freshness.live===true);
  const freshnessLabel=freshness&&freshness.live===true?(current?(freshnessAge===null?"fresh · age unknown":"fresh · "+fmtDuration(freshnessAge)):"fresh metadata · selected run is not live"):
    freshness&&freshness.live===false?(freshnessAge===null?"stale · age unknown":"stale · "+fmtDuration(freshnessAge)):
    freshnessAge===null?"unknown · BEPA freshness unavailable":"unknown · snapshot age "+fmtDuration(freshnessAge);
  const liveBepa=current&&current.bepa||{};
  const activeTeacherJobs=teacherFresh?safeArray(liveBepa.active_teacher_jobs):[];
  const configuredTeacher=workspace.source&&workspace.source.runtime&&workspace.source.runtime.bepa&&workspace.source.runtime.bepa.teacher||{};
  const teacherJobsTable=!teacherFresh?emptyState("活动教师任务未知",freshnessLabel+"；未把缺失或过期的 active_teacher_jobs 当作空队列。"):
    activeTeacherJobs.length?dataTable("bepa-live-teacher-jobs",activeTeacherJobs,[
    {key:"job_id",label:"活动任务",render:(row)=>el("span",{className:"mono"},compact(row.job_id,22))},
    {key:"task_id",label:"Task"},
    {key:"state",label:"台账状态",render:(row)=>badge(row.state)},
    {key:"lease_expires_at",label:"租约",render:(row)=>typeof row.lease_expires_at!=="number"?badge("未知","warn"):row.lease_expires_at<=Date.now()/1000?badge("已到期，状态待确认","warn"):fmtDate(row.lease_expires_at)},
    {key:"tool_calls",label:"工具调用",numeric:true},
    {key:"confirmed_uncached_input_tokens",label:"已确认未快取输入",numeric:true,render:(row)=>fmtInt(row.confirmed_uncached_input_tokens)},
    {key:"reserved_tokens",label:"预留 tokens",numeric:true,render:(row)=>fmtInt(row.reserved_tokens)},
    {key:"source_verified",label:"来源验证",render:(row)=>row.source_verified===true?badge("verified","good"):badge("pending","warn")}
  ],{search:false,pageSize:5}):emptyState("当前没有活动教师任务","BEPA freshness 为新鲜；历史任务见下方 job 台账。");
  const teacherPanel=panel("Teacher 即时状态与配置",el("div",{className:"drawer-stack"},
    kvRows([
      ["中央维护模式",teacherFresh?liveBepa.mode:null],
      ["活动任务",teacherFresh?activeTeacherJobs.length:null],
      ["观测模型",teacherFresh?liveBepa.teacher_model:null],
      ["当前预算 scope",teacherFresh?liveBepa.current_budget_scope:null],
      ["最后领取轮询",teacherFresh&&liveBepa.teacher_last_seen?ago(liveBepa.teacher_last_seen):null],
      ["配置模型",configuredTeacher.model],
      ["配置并发上限",configuredTeacher.max_active_jobs],
      ["BEPA 即时 freshness",freshnessLabel],
      ["观测状态",teacherFresh?"当前 BEPA metadata":"即时状态未知；台账为持久证据"]
    ]),teacherJobsTable),{eyebrow:"LIVE / CONFIGURED SEPARATELY",description:"任务来自BEPA台账；领取轮询不是任务心跳，中央idle也不代表Teacher没有工作。"});
  if(!bepa.available)return {element:el("div",{className:"view-stack"},el("header",{className:"section-title"},el("div",{className:"eyebrow"},"BEPA / TEACHER ASSISTED"),el("h2",{},"BEPA"),el("p",{},"Teacher traces、verified plans、guided reruns 与 durable reservations。")),el("div",{className:"notice notice-info"},el("span",{className:"notice-mark"},"i"),el("div",{},el("b",{},"BEPA ledger unavailable"),el("p",{},bepa.reason||"当前来源未提供可读台账。"))),bepa.source?el("p",{className:"file-path"},bepa.source):null,teacherLogs.element),mount:teacherLogs.mount};
  const scopes=safeArray(bepa.scopes);
  const scope=bepa.scope&&typeof bepa.scope==="object"?bepa.scope:null;
  const jobs=safeArray(bepa.jobs),plans=safeArray(bepa.plans),attempts=safeArray(bepa.attempts),reservations=safeArray(bepa.reservation_ledger),cacheRows=safeArray(bepa.cache_entries),cacheSummaryData=bepa.cache&&typeof bepa.cache==="object"&&!Array.isArray(bepa.cache)?bepa.cache:null,timeline=safeArray(bepa.timeline),pipeline=safeArray(bepa.pipeline);
  const activeJobById=new Map(activeTeacherJobs.filter((item)=>item&&item.job_id).map((item)=>[item.job_id,item]));
  const jobCount=scope&&typeof scope.job_count==="number"?scope.job_count:(Array.isArray(bepa.jobs)?jobs.length:null);
  const planCount=pipelineEntry(bepa,"plans")&&typeof pipelineEntry(bepa,"plans").count==="number"?pipelineEntry(bepa,"plans").count:(Array.isArray(bepa.plans)?plans.length:null);
  const cacheCount=cacheSummaryData&&typeof cacheSummaryData.metadata_candidates==="number"?cacheSummaryData.metadata_candidates:(Array.isArray(bepa.cache_entries)?cacheRows.length:null);
  const stageList=timeline.length?el("div",{className:"timeline-overview"},timeline.slice(0,5).map((item)=>el("button",{className:"timeline-task-card",type:"button",onClick:()=>timelineDrawer(item,runId)},el("div",{className:"timeline-task-head"},el("b",{className:"mono"},compact(item.task_key,30)),item.wait&&item.wait.elapsed_seconds!==null&&item.wait.elapsed_seconds!==undefined?badge("wait "+fmtDuration(item.wait.elapsed_seconds),"warn"):badge("age unknown")),stageTokens(item.stages),item.wait&&item.wait.time_basis?el("small",{className:"muted"},"Wait basis: "+item.wait.time_basis):null))):emptyState("没有等待任务","时间线按任务台账来源生成。");
  const jobTable=dataTable("bepa-jobs",jobs,[
    {key:"job_id",label:"Teacher job",render:(row)=>el("span",{className:"mono"},compact(row.job_id,22))},
    {key:"state",label:"状态",render:(row)=>badge(row.state)},
    {key:"budget_scope",label:"Budget scope",render:(row)=>compact(row.budget_scope,22)},
    {key:"used_tokens",label:"已结算 tokens",numeric:true,render:(row)=>row.used_tokens===null||row.used_tokens===undefined?DASH:fmtInt(row.used_tokens)},
    {key:"live_tokens",label:"未快取输入 tokens",numeric:true,sortValue:(row)=>activeJobById.get(row.job_id)?.confirmed_uncached_input_tokens,render:(row)=>{const active=activeJobById.get(row.job_id);return active?fmtInt(active.confirmed_uncached_input_tokens):DASH;}},
    {key:"reserved_tokens",label:"预留 tokens",numeric:true,render:(row)=>row.reserved_tokens===null||row.reserved_tokens===undefined?DASH:fmtInt(row.reserved_tokens)},
    {key:"max_tokens",label:"单任务上限",numeric:true},
    {key:"attempts",label:"Attempts",numeric:true,render:(row)=>row.attempts===null||row.attempts===undefined?DASH:fmtInt(row.attempts)},
    {key:"fail_reason",label:"Reason"},
    {key:"plan_artifact_ref",label:"Plan",sortable:false,render:(row)=>artifactButton(row.plan_artifact_ref,"Read plan",runId)},
    {key:"updated_at",label:"Updated",render:(row)=>row.updated_at?ago(row.updated_at):DASH},
  ],{pageSize:10,note:sectionNote(bepa,"jobs","Settled usage comes from the ledger; active provider receipts and reserved grants are shown separately")});
  const jobsContent=jobs.length?jobTable:(Array.isArray(bepa.jobs)&&jobCount===0?emptyState("No Teacher jobs in selected scope",sectionNote(bepa,"jobs")):statusSummary("Teacher job states unavailable",(scope&&scope.jobs_by_state)||((pipelineEntry(bepa,"teacher_jobs")||{}).by_state),bepa.reason||"Selected-scope job rows are unavailable."));
  const planRows=plans;
  const planContent=plans.length?null:(Array.isArray(bepa.plans)&&planCount===0?emptyState("No plans in selected scope",sectionNote(bepa,"plans")):emptyState("Plan row details unavailable",bepa.reason||"The selected BEPA source did not include plan artifact rows."));
  const planTable=planRows.length?dataTable("bepa-plans",planRows,[
    {key:"plan_id",label:"Plan",render:(row)=>el("span",{className:"mono"},compact(row.plan_id,22))},
    {key:"task_key",label:"Task",render:(row)=>el("span",{className:"mono"},compact(row.task_key,24))},
    {key:"status",label:"状态",render:(row)=>badge(row.status)},
    {key:"seed_attempts",label:"Guided runs",numeric:true},
    {key:"plan_artifact_ref",label:"Teacher plan",sortable:false,render:(row)=>artifactButton(row.plan_artifact_ref,"Read plan",runId)},
    {key:"source_receipt_ref",label:"Source receipt",sortable:false,render:(row)=>artifactButton(row.source_receipt_ref,"Teacher receipt",runId)},
    {key:"updated_at",label:"Updated",render:(row)=>row.updated_at?ago(row.updated_at):DASH},
  ],{pageSize:10,note:sectionNote(bepa,"plans","A ready plan remains reusable after a guided cache is produced")}):emptyState("Plan row details unavailable","Task timeline carries plan status and ID; the selected-scope plan row list was not included.");
  const attemptTable=attempts.length?dataTable("bepa-attempts",attempts,[
    {key:"task_key",label:"Task",render:(row)=>compact(row.task_key,26)},
    {key:"job_id",label:"Job",render:(row)=>compact(row.job_id,20)},
    {key:"attempt_id",label:"Attempt",render:(row)=>compact(row.attempt_id,20)},
    {key:"status",label:"状态",render:(row)=>badge(row.status)},
    {key:"reason",label:"Reason"},
    {key:"updated_at",label:"Updated",render:(row)=>row.updated_at?ago(row.updated_at):DASH},
  ],{pageSize:10,note:sectionNote(bepa,"attempts","Persisted seed_attempt rows")}):emptyState("No attempt rows","No selected-scope seed attempt rows were returned.");
  const attemptContent=attempts.length?attemptTable:(Array.isArray(bepa.attempts)&&attempts.length===0?emptyState("No guided attempts in selected scope",sectionNote(bepa,"attempts")):emptyState("Attempt rows unavailable",bepa.reason||"No selected-scope seed attempt rows were returned."));
  const reservationTable=reservations.length?dataTable("bepa-reservations",reservations,[
    {key:"task_key",label:"Task",render:(row)=>compact(row.task_key,24)},
    {key:"group_id",label:"Group",render:(row)=>compact(row.group_id,22)},
    {key:"state",label:"State",render:(row)=>badge(row.state)},
    {key:"cache_entry_id",label:"Cache entry",render:(row)=>compact(row.cache_entry_id,20)},
    {key:"engine_epoch",label:"Engine epoch",render:(row)=>compact(row.engine_epoch,18)},
    {key:"update_attempt_id",label:"Update attempt",render:(row)=>compact(row.update_attempt_id,18)},
    {key:"created_at",label:"Created",render:(row)=>row.created_at?ago(row.created_at):DASH},
  ],{pageSize:10,note:sectionNote(bepa,"reservation_ledger","Reservation ledger outcome is the commit source; cache counters remain separate")}):emptyState("No reservation rows","No selected task reservation rows were returned.");
  const reservationContent=reservations.length?reservationTable:(Array.isArray(bepa.reservation_ledger)&&reservations.length===0?emptyState("No reservations for observed tasks",sectionNote(bepa,"reservation_ledger")):emptyState("Reservation rows unavailable",bepa.reason||"No selected task reservation rows were returned."));
  const cacheTable=cacheRows.length?dataTable("bepa-cache",cacheRows,[
    {key:"cache_entry_id",label:"Entry",render:(row)=>el("span",{className:"mono"},compact(row.cache_entry_id,20))},
    {key:"task_key",label:"Task",render:(row)=>el("span",{className:"mono"},compact(row.task_key,24))},
    {key:"row_kind",label:"Kind",render:(row)=>badge(row.row_kind||"unknown")},
    {key:"active",label:"Active",render:(row)=>row.active?badge("active","good"):badge("inactive","neutral")},
    {key:"consumability",label:"Admission",render:(row)=>row.consumability||"unknown"},
    {key:"central_claim_receipt_recheck",label:"Receipt check",render:(row)=>row.central_claim_receipt_recheck||"not performed"},
    {key:"committed_uses",label:"Committed uses",numeric:true},
    {key:"source_policy_version",label:"Source policy",numeric:true},
    {key:"raw_artifact_ref",label:"Raw episode",sortable:false,render:(row)=>artifactButton(row.raw_artifact_ref,String(row.row_kind||"").includes("guided")?"Guided raw":"Raw episode",runId)},
    {key:"projection_artifact_ref",label:"Training projection",sortable:false,render:(row)=>artifactButton(row.projection_artifact_ref,"Projection",runId)},
    {key:"receipt_artifact_ref",label:"Receipt",sortable:false,render:(row)=>artifactButton(row.receipt_artifact_ref,String(row.row_kind||"").includes("guided")?"Guided receipt":"Source receipt",runId)},
    {key:"created_at",label:"Created",render:(row)=>row.created_at?ago(row.created_at):DASH},
  ],{pageSize:10,note:sectionNote(bepa,"cache_entries","Metadata candidate ≠ verified learner admission")}):cacheSummary(cacheSummaryData,timeline);
  return {element:el("div",{className:"view-stack"},
    el("header",{className:"section-title"},el("div",{className:"eyebrow"},"TEACHER-ASSISTED EVIDENCE"),el("h2",{},"BEPA"),el("p",{},"预算 scope、任务阶段、引导计划、student cache 与 optimizer reservation 分开呈现。")),
    teacherPanel,
    teacherLogs.element,
    grid([metric("Budget scope",scopes.length?fmtInt(scopes.length):(scope&&scope.available===true?"0":DASH),"scopes",bepa.source||"scope source unknown"),metric("Teacher jobs",fmtInt(jobCount),"jobs","selected scope · persisted rows"),metric("Guided plans",fmtInt(planCount),"plans","selected scope pipeline count"),metric("Cache metadata",fmtInt(cacheCount),"candidates","consumability remains unknown")],"four"),
    panel("预算 scopes",scopes.length?el("div",{className:"scope-grid"},scopes.map(scopeCard)):emptyState("Configured budget scope unavailable",(scope&&scope.available===false&&scope.reason)||bepa.reason||"The selected runtime did not provide a readable budget scope."),{eyebrow:"TOKEN BUDGET",description:sectionNote(bepa,"scopes","Used、reserved 与 usage unknown 保持独立；job grant 不等于完整费用。")}),
    panel("近期任务阶段",stageList,{eyebrow:"NATIVE → TEACHER → PLAN → GUIDED → CACHE → COMMIT",description:"Plan 在 guided cache 创建后仍可复用；metadata 候选不代表通过 provenance 重验。"}),
    panel("全部任务时间线",timelineTable(timeline,runId),{eyebrow:"TASK TIMELINE INDEX",description:"可搜索、排序和分页；等待年龄与时间依据仅按记录来源显示。",flush:true}),
    panel("Teacher jobs",jobsContent,{eyebrow:"PROVIDER RESPONSES",flush:true}),
    grid([panel("Plans",plans.length?planTable:planContent,{eyebrow:"VERIFIED PLAN",flush:true}),panel("Student cache",cacheRows.length?cacheTable:cacheSummary(cacheSummaryData,timeline),{eyebrow:"GUIDED / NATIVE CACHE",flush:true})],"two"),
    grid([panel("Guided attempts",attemptContent,{eyebrow:"PERSISTED SEED ATTEMPTS",flush:true}),panel("Reservations and commits",reservationContent,{eyebrow:"DURABLE USES",flush:true})],"two"),
    panel("All-scope history",el("div",{className:"drawer-stack"},statusSummary("Historical state counts unavailable",(bepa.historical||{}).job_states_all_scopes,"No all-scope job status counts were supplied."),statusSummary("Reservation counts unavailable",(bepa.historical||{}).reservation_states_all_scopes,"No all-scope reservation status counts were supplied.")),{eyebrow:"GLOBAL LEDGER COUNTS",description:"Historical counts are separate from current scope and from the selected run cohort."}),
    panel("预算与台账来源",el("div",{className:"drawer-stack"},el("div",{className:"evidence-note evidence-warn"},el("strong",{},"Guided evidence has an influence path"),el("p",{},"教师方案引导的 Qwen rerun 与自然策略证据分开统计。Plan 在 cache 产生后仍可保留并再次使用。")),jsonDetails("Source roots",bepa.candidates||bepa.source||{})),{eyebrow:"PROVENANCE"})
  ),mount:teacherLogs.mount};
}
