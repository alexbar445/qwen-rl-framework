import {api,asText,compact,el,fmtDate,jsonDetails,openDrawer,kvRows,panel,path,safeArray,sourceNote,makeLink} from "../core.js";
export function selectedRun(workspace){return workspace&&workspace.run||{};}
export function records(workspace){return safeArray(selectedRun(workspace).records);}
export function latestRecord(workspace){const items=records(workspace);return items.length?items[items.length-1]:null;}
export function liveSnapshot(workspace){
  if(!workspace||!workspace.selection||workspace.selection.live_selected!==true)return null;
  const watch=workspace.watch||{};
  if(watch.ok!==true||!watch.snapshot)return null;
  // Hardware belongs to this selected workspace, not the no-GPU metadata sampler.
  return {...watch.snapshot,gpus:Array.isArray(workspace.gpus)?workspace.gpus:[]};
}
export function historical(workspace){return !!(workspace&&workspace.selection&&workspace.selection.view_mode==="history");}
export function groups(workspace){
  const out=[];
  for(const record of records(workspace))for(const item of safeArray(record.groups))out.push(Object.assign({record_id:record.record_id,record_status:record.status,record_at:record.finished_at||record.started_at},item));
  return out;
}
export function openRecord(record){
  openDrawer("更新记录 "+asText(record.record_id),el("div",{className:"drawer-stack"},
    panel("记录摘要",kvRows([
      ["状态",record.status],["阶段",record.kind],["策略版本",record.policy_version],["更新数",record.updates],
      ["请求步数",record.requested_steps],["耗时",record.elapsed_seconds===null?null:(Number(record.elapsed_seconds).toFixed(1)+" s")],
      ["开始",fmtDate(record.started_at)],["结束",fmtDate(record.finished_at)],["错误",record.error],
    ]),{eyebrow:"RUN RECORD"}),
    jsonDetails("训练 summary",record.summary||{}),
    jsonDetails("metrics",record.metrics||{}),
    jsonDetails("训练进度",record.progress||{}),
    jsonDetails("policy handoffs",record.policy_handoffs||[]),
    jsonDetails("BEPA seed observability",record.bepa_seed_observability||{})));
}
export async function openArtifact(ref,runId,label="产物"){
  try{
    const result=await api("/api/artifact",{ref,run_id:runId});
    const payload=result.payload||result;
    const guided=/plan|guided|teacher|projection/i.test(String(label))||String(label).includes("计划");
    openDrawer(guided?"教师计划 · 引导轨迹":label,el("div",{className:"drawer-stack"},
      guided?el("div",{className:"evidence-note evidence-warn"},el("strong",{},"Teacher-guided evidence"),el("p",{},"该计划会影响学生的执行路径，缓存回放样本与模型原生轨迹分开统计。")):null,
      kvRows([["引用",ref],["run",runId||"全局 BEPA store"],["文件字节",result.bytes]]),
      jsonDetails("产物内容",payload,48000)),guided?"BEPA / guided bias":"ARTIFACT");
  }catch(error){openDrawer(label,el("div",{className:"error-block"},"无法读取该产物：",error.message),"READ ERROR");}
}
export async function openTrajectory(runId,name){
  try{
    const data=await api("/api/trajectory",{run_id:runId,name});
    openDrawer("轨迹组 "+compact(data.group_id||name,40),el("div",{className:"drawer-stack"},
      kvRows([["文件",data.file],["Group",data.group_id],["Phase",data.phase],["策略版本",data.policy_version],["Adapter SHA",data.adapter_sha256],["Schema",data.schema]]),
      panel("Episode rows",el("div",{className:"episode-list"},safeArray(data.rows).map((row,index)=>el("article",{className:"episode-card"},
        el("div",{className:"episode-head"},el("b",{},"Episode "+(index+1)),el("span",{className:"mono"},compact(row.episode_id,32)),row.row_kind?el("span",{className:"tag"},row.row_kind):null),
        kvRows([["Worker",row.worker_id],["Policy",row.policy_version],["Reward",row.reward],["Native exact",row.exact===true?"是":row.exact===false?"否":null],["输入 tokens",row.input_tokens],["生成 tokens",row.generated_tokens],["观察 tokens",row.observation_tokens],["Tool events",row.tool_events],["Cost",row.cost]]),
        jsonDetails("消息结构（内容已省略）",row.messages||[])))),{eyebrow:"TRAJECTORY FILE"}),
      sourceNote("来源：run.artifacts.trajectories")));
  }catch(error){openDrawer(name,el("div",{className:"error-block"},"轨迹读取失败：",error.message),"READ ERROR");}
}
export function eventDrawer(item){
  const severity=item.severity===null||item.severity===undefined||String(item.severity).trim()===""?"未记录":
    String(item.severity).toLowerCase()==="unknown"?"未知":item.severity;
  const version=item.policy_version;
  const policy=typeof version==="number"&&Number.isInteger(version)&&version>=0?"v"+String(version):
    item.policy_applicability==="not_applicable"?"不适用":"未记录";
  const applicability=item.policy_applicability==="known"?"已知":
    item.policy_applicability==="not_applicable"?"不适用":item.policy_applicability==="unknown"?"未记录":"未记录";
  const detail=item.detail===null||item.detail===undefined?
    panel("事件详情",el("p",{className:"body-copy"},"未记录")):
    typeof item.detail==="string"?panel("事件详情",el("p",{className:"body-copy"},item.detail)):jsonDetails("事件 detail",item.detail);
  const evidenceScope=item.evidence_scope==="shared_bepa_ledger"?
    "共享 BEPA 事件台账；可能跨 run，不代表当前 run 独占":item.evidence_scope;
  openDrawer(item.title||item.type||"事件",el("div",{className:"drawer-stack"},
    kvRows([["时间",fmtDate(item.at)],["类型",item.type],["级别",severity],["Severity basis",item.severity_basis],["来源",item.source],["引用",item.ref],
      ["Task key",item.task_key],["Group",item.group_id],["Policy",policy],["Policy applicability",applicability],
      ["Policy role",item.policy_role],["Policy basis",item.policy_basis],["Budget scope",item.budget_scope],
      ["Engine epoch",item.engine_epoch],["Evidence scope",evidenceScope],["Action",item.action],
      ["Reason",item.reason??"未附加原因"],["Detail status",item.detail_status],["Sequence",item.sequence],["Entity ID",item.entity_id]]),
    item.message?panel("事件说明",el("p",{className:"body-copy"},item.message)):null,
    detail,
    jsonDetails("关联 evidence",item.evidence||[]),
    jsonDetails("原始 event payload",item)));
}
export function idColumn(value){return el("span",{className:"mono"},compact(value,30));}
export function runSource(workspace,label="selected run"){return sourceNote("来源："+label+(selectedRun(workspace).run_id?" · "+selectedRun(workspace).run_id:""));}
