import {executorRates} from "./executor-rates.js";
import {DASH,ago,badge,dataTable,el,emptyState,firstNumber,fmt,fmtDuration,fmtInt,fmtPct,grid,jsonDetails,metric,panel,safeArray,sectionTitle,sourceNote} from "../core.js";
import {liveSnapshot,latestRecord,openRecord,records,selectedRun} from "./common.js";
import {mountHistoryChart} from "../charts.js";

function recordRows(items){return items.slice().reverse();}
function objectRows(object){return Object.entries(object||{}).filter(([,v])=>v!==undefined).map(([name,value])=>({name,value}));}
function summaryValue(record,keys){for(const key of keys){const value=record&&record.summary&&record.summary[key];if(value!==null&&value!==undefined)return value;}return null;}
export function renderLearner(workspace){
  const snapshot=liveSnapshot(workspace),run=selectedRun(workspace),record=latestRecord(workspace);
  const learner=snapshot&&snapshot.learner||{},training=learner.training||{},progress=training.progress||{},quality=training.quality||{};
  const lifecycle=snapshot&&snapshot.lifecycle||{};
  const centralSourceBound=Boolean(snapshot&&snapshot.owned_source_verified===true&&lifecycle.backend==="central_gradient");
  const centralUpdateActive=Boolean(centralSourceBound&&training.active_update&&typeof training.active_update==="object"&&!Array.isArray(training.active_update));
  const activity=workspace.activity||{};
  const throughput=snapshot&&snapshot.throughput||{},committed=snapshot&&snapshot.committed||{},runCommit=committed.run||{},process=committed.process||{};
  const rolling=runCommit.rolling||{},rates=rolling.rates_per_hour||{};
  const forwardScope=throughput.learner_forward_scope||null;
  const partial=forwardScope==="rank0_partial"||String(forwardScope||"").includes("rank0_partial");
  const centralForwardUnknown=centralSourceBound&&forwardScope==="not_reported";
  const warning=!snapshot?(workspace.selection&&workspace.selection.live_selected===false?"历史 run 中没有当前 Learner 快照。":(workspace.watch&&workspace.watch.error)||"实时 Learner 状态未知。"):
    partial?"DP 训练当前只提供 rank0 前向计数；不可按 DP 全局吞吐解读。":
    centralForwardUnknown?"中央未回报聚合前向计数；各 executor 的 CUDA 实测速率见下方。":"";
  const phase=centralSourceBound?(activity.phase||(centralUpdateActive?"梯度/优化器更新（子阶段未知）":null)):
    progress.stage||progress.phase||learner.activity_state||null;
  const phaseElapsed=centralSourceBound?activity.elapsed_seconds:progress.stage_elapsed_seconds;
  const micro=centralSourceBound?null:progress.current_microbatch;
  const planned=centralSourceBound?null:progress.planned_microbatches;
  const microValue=centralUpdateActive?"未知":(micro!==null&&micro!==undefined?(fmtInt(micro)+" / "+fmtInt(planned)):DASH);
  const microMeta=centralUpdateActive?"中央 monitor 未上报 executor 微批进度；原始 progress 保留在下方证据中":
    centralSourceBound?"当前没有可用的 executor 微批进度":"计划与完成进度分开";
  const forwardScopeLabel=centralForwardUnknown?"中央 gradient executor 前向计数未上报；各卡实测见下方":forwardScope||"measurement scope 未提供";
  const phaseReportedStage=centralSourceBound?activity.reported_stage:null;
  const updateMetrics=record&&record.metrics||{};
  const chart=el("div",{className:"history-chart"});
  const commitTotals=runCommit.totals||{};
  const tiles=grid([
    metric("前台阶段",phase||DASH,"",phaseElapsed!==null&&phaseElapsed!==undefined?
      fmtDuration(phaseElapsed)+(centralUpdateActive?" 本次更新 wall-time":" 当前阶段已用"):
      (snapshot?(centralUpdateActive?"更新 wall-time 未提供":"阶段 elapsed 未提供"):"历史记录未知"),"teal"),
    metric("微批进度",microValue,centralSourceBound?"":"microbatches",
      centralSourceBound?microMeta:(progress.trajectories!==undefined?fmtInt(progress.trajectories)+" 条序列":microMeta)),
    metric("Learner 前向",fmt(throughput.learner_forward_input_tokens_per_second),"tok/s",forwardScopeLabel,"blue"),
    metric("Run 提交更新",fmtInt(firstNumber(commitTotals.updates,runCommit.logical_completed_updates,record&&record.updates)),"updates",runCommit.scope||"run record",""),
    metric("Run 已提交 input",fmtInt(commitTotals.input_tokens),"tokens","只计 app 确认的 optimizer commit","amber"),
    metric("Run 已提交 policy",fmtInt(commitTotals.policy_tokens),"tokens","学习目标 token；与生成输出分开","ink"),
  ],"three");
  const phaseBody=snapshot?el("div",{className:"stage-board"},
    el("div",{className:"stage-current"},badge(phase||"unknown"),el("span",{},phase||progress.stage||"等待 Learner telemetry"),
      phaseElapsed!==null&&phaseElapsed!==undefined?el("small",{},fmtDuration(phaseElapsed)+(centralUpdateActive?" 本次更新 wall-time":" 当前阶段已用")):null,
      centralSourceBound&&phaseReportedStage?el("small",{},"原始上报 stage："+String(phaseReportedStage)):null),
    el("div",{className:"stage-grid"},Object.entries(progress.stage_seconds||{}).map(([key,value])=>el("div",{className:"stage-time"},el("span",{},key),el("b",{},fmtDuration(value))))),
    jsonDetails("完整进度证据",progress)):emptyState("没有实时进度",warning);
  const qualityRows=objectRows(quality);
  const correction=quality.importance_correction||{};
  const alignment=quality.alignment||{};
  const clip=quality.clip_ratio||{};
  const qualitySummary=quality.available===true?grid([
    metric("Latest loss",fmt(quality.loss,5),"",quality.source||"last successful optimizer update","teal"),
    metric("Gradient norm",fmt(firstNumber(quality.gradient_norm_before_clip,quality.gradient_norm),5),"",quality.gradient_norm_after_clip!==undefined?"after clip "+fmt(quality.gradient_norm_after_clip,5):"pre-clip norm","blue"),
    metric("Learning rate",fmt(quality.learning_rate,8),"", "optimizer group 0",""),
    metric("Importance cap",fmt(correction.cap??quality.importance_sampling_cap,3),"x",correction.truncated_fraction!==undefined?fmtPct(100*correction.truncated_fraction,2)+" of fresh tokens capped":"capped fraction unknown","amber"),
  ],"four"):emptyState("Latest update quality unavailable","No successful optimizer update quality snapshot was reported.");
  const clipRows=Object.entries(clip).map(([name,value])=>({name,value}));
  const alignText=alignment.status==="policy_lag"?"Policy lag "+(alignment.policy_lag??"unknown")+"; no same-policy numeric comparison":alignment.status?alignment.status+" · mean "+fmt(alignment.mean_abs_error,5)+" · RMS "+fmt(alignment.rms_error,5)+" · max "+fmt(alignment.max_abs_error,5):"Unknown";
  const health=snapshot&&snapshot.learner_health||{};
  const source=workspace.source||{},runtime=source.runtime||{},runtimeLearner=runtime.learner||{},runtimeQuant=runtime.quantization||{};
  const compute=health.compute||{},healthEvidence=health.evidence||{},actualQuant=health.quantization||healthEvidence.quantization||{};
  const actualDevices=compute.devices||snapshot&&snapshot.learner_gpus||null;
  const configuredGpuList=Array.isArray(runtimeLearner.gpus)?runtimeLearner.gpus.join(", "):null;
  const actualGPUList=Array.isArray(actualDevices)?actualDevices.map((device)=>typeof device==="object"?(device.id??device.device??JSON.stringify(device)):device).join(", "):null;
  const capability=panel("Loaded runtime capability",grid([
    panel("Loaded /health",el("div",{className:"capability-kv"},
      capabilityPair("Compute state",compute.state),capabilityPair("Phase",compute.phase),capabilityPair("Mode",compute.mode||compute.topology&&compute.topology.mode),capabilityPair("World size",compute.world_size),capabilityPair("Devices",actualGPUList),capabilityPair("Quantization",actualQuant.mode),capabilityPair("Quant method",actualQuant.quant_method),capabilityPair("Bits",actualQuant.bits),capabilityPair("Max sequence length",health.max_sequence_length),capabilityPair("Padding side",healthEvidence.padding_side||health.padding_side),capabilityPair("Pad token id",healthEvidence.pad_token_id||health.pad_token_id),jsonDetails("Rank evidence",compute.rank_evidence||compute.topology||{}),jsonDetails("Loaded health evidence",healthEvidence)),{eyebrow:"ACTUAL PROCESS"}),
    panel("Selected runtime config",el("div",{className:"capability-kv"},
      capabilityPair("Parallelism",runtimeLearner.parallelism),capabilityPair("Learner GPUs",configuredGpuList),capabilityPair("Quantization mode",runtimeQuant.mode),capabilityPair("Quant bits",runtimeQuant.bits),capabilityPair("Max sequence length",runtimeLearner.max_sequence_length),capabilityPair("Microbatch size",runtimeLearner.microbatch_size),capabilityPair("Groups / update",runtimeLearner.groups_per_update),capabilityPair("Padding side",runtimeLearner.padding_side||runtime.model&&runtime.model.padding_side),jsonDetails("Selected runtime",runtime)),{eyebrow:"CONFIGURED SOURCE"})
  ],"two"),{eyebrow:"RUNTIME IDENTITY",description:"Loaded /health values and selected source config are kept in separate columns. Missing rank, padding or length evidence stays unknown."});
  const recordList=recordRows(records(workspace));
  const recordTable=dataTable("learner-records",recordList,[
    {key:"record_id",label:"记录",render:(row)=>row.record_id||DASH},
    {key:"status",label:"状态",render:(row)=>badge(row.status)},
    {key:"kind",label:"类型"},
    {key:"updates",label:"更新",numeric:true},
    {key:"policy_version",label:"策略版本",numeric:true},
    {key:"elapsed_seconds",label:"耗时",numeric:true,render:(row)=>fmtDuration(row.elapsed_seconds)},
    {key:"finished_at",label:"完成",render:(row)=>row.finished_at?ago(row.finished_at):DASH},
  ],{pageSize:10,note:"点击记录查看其原始 summary 与 metrics",onRow:openRecord});
  const output=el("div",{className:"view-stack"},
    sectionTitle("LEARNER","Learner","训练前向、更新阶段、实际提交计数与最近优化品质。"),
    warning?el("div",{className:"notice "+(partial?"notice-warn":"notice-info")},el("span",{className:"notice-mark"},partial?"!":"i"),el("div",{},el("b",{},partial?"局部观测":"状态说明"),el("p",{},warning))):null,
    tiles,
    capability,
    centralSourceBound?panel("Executor 前向 / 反向实测",executorRates(snapshot,true),{eyebrow:"COMPLETED CUDA MICRO-BATCH",description:"每个 executor 分别显示最近完成的 CUDA 微批，不汇总成虚构的全机瞬时速率。"}):null,
    grid([
      panel("实时更新阶段",phaseBody,{eyebrow:"TRAINING PROGRESS",description:"计划工作量和已完成工作分开呈现。",meta:sourceNote(snapshot?(centralSourceBound?"workspace.activity + source-bound central watch":"watch.learner.training.progress"):"selected run records")}),
      panel("最近有效更新品质",el("div",{className:"drawer-stack"},qualitySummary,clipRows.length?dataTable("learner-clip-ratio",clipRows,[{key:"name",label:"TRL clip ratio"},{key:"value",label:"观测值",render:(row)=>fmt(row.value,5)}],{search:false,pageSize:10}):null,el("div",{className:"alignment-summary"},el("span",{className:"eyebrow"},"LOGPROB ALIGNMENT"),el("b",{},alignText),alignment.scope?el("small",{},alignment.scope):null),jsonDetails("Quality fields",quality)),{eyebrow:"LATEST COMMITTED UPDATE",meta:sourceNote(snapshot?"watch.learner.training.quality":"run record.metrics")})
    ],"two"),
    panel("提交账本",grid([
      ledgerCard("当前进程 run",runCommit,fmtInt),
      ledgerCard("当前进程累计",process,fmtInt)
    ],"two"),{eyebrow:"APP COMMIT EVIDENCE",description:"恢复前的 token 总量若未知，保持未知；总量不等于每秒吞吐。"}),
    panel("吞吐历史",chart,{eyebrow:"ROLLING WINDOW",description:"Native 在途解码、已校验输出、前向和成功提交分列；未知或采样中断处断线。"}),
    panel("训练记录",recordTable,{eyebrow:"RUN HISTORY",description:run.run_id||"所选 run 未提供 ID",flush:true}),
    record?panel("最近记录原始数值",jsonDetails("summary",record.summary||{}),{eyebrow:"RECORD SNAPSHOT"}):null
  );
  return {element:output,mount:()=>window.echarts&&chart.isConnected?mountHistoryChart(chart,workspace.history||[],document.documentElement.dataset.theme==="dark"):null};
}
function capabilityPair(label,value){return el("div",{className:"capability-line"},el("span",{},label),el("b",{className:/GPU|Devices|token/i.test(label)?"mono":""},value===null||value===undefined||value===""?DASH:(Array.isArray(value)?value.join(", "):String(value))));}
function ledgerCard(label,entry,format){
  const totals=entry.totals||{},rates=entry.rolling&&entry.rolling.rates_per_hour||{},scope=entry.scope||"";
  return panel(label,el("div",{className:"ledger-grid"},
    ["updates","input_tokens","policy_tokens","trajectories"].map((key)=>el("div",{className:"ledger-cell"},el("span",{},key.replaceAll("_"," ")),el("b",{},format(totals[key])),el("small",{},rates[key]===null||rates[key]===undefined?"速率未知":fmt(rates[key]/3600,2)+" /s"))),
    el("div",{className:"ledger-foot"},badge(entry.live===true?"live":entry.live===false?"stale":"unknown"),el("span",{},scope||"scope 未提供"))),{eyebrow:"DURABLE COUNTERS"});
}
