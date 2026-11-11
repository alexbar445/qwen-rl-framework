import {renderEvaluationOverview} from "./evaluation.js";
import {executorRates} from "./executor-rates.js";
import {DASH,ago,badge,el,emptyState,fmt,fmtDuration,fmtInt,grid,metric,panel,safeArray,sourceNote,compact,openDrawer,jsonDetails,kvRows} from "../core.js";
import {liveSnapshot,latestRecord,selectedRun,eventDrawer} from "./common.js";
import {mountHistoryChart} from "../charts.js";
import {workerEvidence} from "./worker-evidence.js";

const MAX_WATCH_AGE_SECONDS=60;
function watchWithinThreshold(workspace){
  const age=workspace.watch&&workspace.watch.age_seconds;
  return typeof age!=="number"||age<=MAX_WATCH_AGE_SECONDS;
}
function liveReason(workspace){
  const watch=workspace.watch||{};
  if(workspace.selection&&workspace.selection.view_mode==="history")return "正在复盘历史 run；实时状态已按 run 隔离";
  if(!watch.ok)return watch.error||"watch 快照不可用";
  if(!watchWithinThreshold(workspace))return "快照已过期 "+Math.round(watch.age_seconds)+" 秒";
  const learnerFresh=watch.snapshot&&watch.snapshot.freshness&&watch.snapshot.freshness.learner;
  if(!watch.snapshot||!learnerFresh||learnerFresh.live!==true){
    const age=learnerFresh&&learnerFresh.age_seconds;
    return typeof age==="number"
      ?"Learner 元数据已过期 "+Math.round(age)+" 秒；当前训练指标未知"
      :"Learner 新鲜度未知；当前训练指标未知";
  }
  return "";
}
function pipelineView(stages,liveSelected){
  const rows=safeArray(stages);
  if(!rows.length)return emptyState(liveSelected===false?"历史 run 不附带实时流水线":"流水线观测未知",liveSelected===false?"当前页只显示所选 run 证据；不会混入其他 run 的 live stages。":"当前 watch 没有可用的阶段计数。");
  return el("div",{className:"pipeline-list"},rows.map((stage,index)=>{
    const count=stage.count===null||stage.count===undefined||stage.known===false?"未知":fmtInt(stage.count)+(stage.unit?" "+stage.unit:"");
    const known=stage.known===false||stage.count===null||stage.count===undefined;
    const state=stage.state|| (known?"unknown":"active");
    return el("div",{className:"pipeline-row"},
      el("div",{className:"pipeline-index"},String(index+1).padStart(2,"0")),
      el("div",{className:"pipeline-main"},el("div",{className:"pipeline-title"},stage.label||stage.id||"阶段",badge(state)),
        el("div",{className:"pipeline-source"},stage.source||"来源未提供",stage.by_state?el("span",{}," · "+Object.entries(stage.by_state).map(([key,value])=>key+" "+value).join(" · ")):null,stage.reason?el("span",{}," · "+stage.reason):null)),
      el("div",{className:"pipeline-count"},count),
      index<rows.length-1?el("div",{className:"pipeline-connector"}):null);
  }));
}
function issueList(insights,events){
  const important=safeArray(insights).slice(0,4);
  if(!important.length)return emptyState("当前没有已归纳的问题","仅显示后端提供的诊断；不从缺失计数推断异常。");
  return el("div",{className:"insight-list"},important.map((item)=>el("button",{className:"insight-item",type:"button",onClick:()=>{
    eventDrawer({title:item.title,message:item.message,detail:item.detail,source:item.source,severity:item.severity,ref:item.ref,evidence:item.evidence});
  }},badge(item.severity),el("span",{className:"insight-copy"},el("b",{},item.title||"诊断"),el("small",{},item.message||item.detail&&typeof item.detail==="string"&&item.detail||"详情未提供")),el("span",{className:"insight-arrow"},"↗"))));
}
function workerSummary(snapshot,watchFresh,source){
  if(!snapshot)return el("div",{className:"quiet-callout"},"当前页没有可用的即时 Worker lease 或 GPU 读数。");
  if(!watchFresh)return el("div",{className:"quiet-callout"},"watch 快照已过期；当前 Worker lease 与 GPU 读数未知。");
  const workers=safeArray(snapshot.workers);
  const sumFresh=(key)=>workers.length&&workers.every((w)=>w.live===true&&typeof w[key]==="number"&&Number.isFinite(w[key])&&w[key]>=0)
    ?workers.reduce((total,w)=>total+w[key],0):null;
  const active=sumFresh("active"),max=sumFresh("maximum"),leases=sumFresh("episode_refs");
  return el("div",{className:"drawer-stack"},el("div",{className:"runtime-strip"},
    el("div",{className:"runtime-stat"},el("span",{},"Worker generation slots"),el("b",{},active===null?DASH:(fmtInt(active)+" / "+fmtInt(max))),el("small",{},"活动请求 / 已知容量")),
    el("div",{className:"runtime-stat"},el("span",{},"Episode references"),el("b",{},leases===null?DASH:fmtInt(leases)),el("small",{},"leases 与 GPU 利用率分开")),
    el("div",{className:"runtime-stat"},el("span",{},"本机 GPU 采样"),el("b",{},safeArray(snapshot.gpus).length?fmtInt(snapshot.gpus.length):DASH),el("small",{},safeArray(snapshot.gpus).length?"nvidia-smi 物理卡读数":"没有有效本机 GPU 读数"))),
    ...workers.map((worker)=>{
      const evidence=workerEvidence(worker,source,safeArray(snapshot.gpus));
      return el("div",{className:"runtime-strip"},
        el("div",{className:"runtime-stat"},el("span",{},worker.id+" · "+evidence.location),el("b",{},worker.live===true?fmtInt(worker.active)+" / "+fmtInt(worker.maximum):DASH),el("small",{},evidence.activity)),
        el("div",{className:"runtime-stat"},el("span",{},"生成 tok/s"),el("b",{},worker.live===true?fmt(worker.tokens_per_second):DASH),el("small",{},"Worker 输出窗口")),
        el("div",{className:"runtime-stat"},el("span",{},"GPU 计算 / 显存"),el("b",{},evidence.gpu?fmt(evidence.gpu.util_percent)+"%":DASH),el("small",{},evidence.gpu?fmtInt(evidence.gpu.memory_used_mib)+" MiB":"对应节点硬件观测未知")));
    }));
}
function timeline(bepa){
  if(!bepa||bepa.available!==true)return emptyState("BEPA 台账不可用","不显示台账不可用时保留的时间线数据。");
  const tasks=safeArray(bepa&&bepa.timeline).slice(0,6);
  if(!tasks.length)return emptyState("暂无任务时间线","该视图只呈现台账记录到的阶段。");
  return el("div",{className:"task-timeline"},tasks.map((task)=>el("button",{className:"task-timeline-row",type:"button",onClick:()=>openDrawer("BEPA task "+(task.task_key||"未知"),el("div",{className:"drawer-stack"},kvRows([["Task",task.task_key],["Current wait",task.wait&&task.wait.stage],["Wait age",task.wait&&task.wait.elapsed_seconds===null?null:task.wait&&fmtDuration(task.wait.elapsed_seconds)],["Time basis",task.wait&&task.wait.time_basis],["Scope",task.scope_id]]),jsonDetails("阶段时间线",task.stages||{})),"BEPA TIMELINE")},
    el("span",{className:"task-key mono"},compact(task.task_key,28)),
    el("span",{className:"timeline-current"},task.wait&&task.wait.stage?task.wait.stage:"阶段未知"),
    el("span",{className:"timeline-age"},task.wait&&task.wait.elapsed_seconds!==null&&task.wait.elapsed_seconds!==undefined?fmtDuration(task.wait.elapsed_seconds):"等待时间未知"),
    el("span",{className:"arrow"},"↗"))));
}
function bepaCoverage(bepa){
  if(!bepa||bepa.available!==true)return emptyState("BEPA 阶段覆盖未知","台账不可用；无法确认当前阶段覆盖。");
  const rows=safeArray(bepa&&bepa.timeline);
  const keys=["native_failure","teacher","plan","guided","cache","reservation","commit"];
  if(!rows.length)return emptyState("BEPA 阶段覆盖未知","没有同一批 task timeline 行，不能构造漏斗计数。");
  return el("div",{className:"bepa-coverage"},
    el("div",{className:"coverage-note"},"计数范围：当前 timeline 的 "+rows.length+" 条 task 行；仅计有事件或阶段状态的行，不是成功率；native_failure 计数表示已记录的失败事件。"),
    el("div",{className:"coverage-grid"},keys.map((key)=>{
      const absent=new Set(["unknown","none","unobserved","not_observed","no_metadata_candidate","not_applicable"]);
      const known=rows.filter((row)=>{
        const state=row.stages&&row.stages[key]&&row.stages[key].state;
        return typeof state==="string"&&!absent.has(state);
      });
      return el("div",{className:"coverage-stage"},el("span",{},key.replaceAll("_"," ")),el("b",{},known.length+" / "+rows.length),el("small",{},known.length?"有事件或阶段状态":"无已知事件或阶段状态"));
    })));
}
export function renderOverview(workspace){
  const snapshot=liveSnapshot(workspace),run=selectedRun(workspace),record=latestRecord(workspace);
  const watchFresh=watchWithinThreshold(workspace);
  const warning=liveReason(workspace),throughput=snapshot&&snapshot.throughput||{},committed=snapshot&&snapshot.committed||{},runCommit=committed.run||{};
  const learnerFresh=Boolean(watchFresh&&snapshot&&snapshot.freshness&&snapshot.freshness.learner&&snapshot.freshness.learner.live===true);
  const workers=safeArray(snapshot&&snapshot.workers);
  const workersFresh=Boolean(watchFresh&&workers.length&&workers.every((worker)=>worker.live===true));
  const outputRate=workersFresh?throughput.aggregate_output_tokens_per_second:null;
  const updates=learnerFresh?pathUpdate(runCommit):(!snapshot&&record?record.updates:null);
  const rateWindow=snapshot&&typeof snapshot.window_seconds==="number"?fmtDuration(snapshot.window_seconds):"窗口未知";
  const workerMetricMeta=workersFresh?(typeof outputRate==="number"&&Number.isFinite(outputRate)?"请求完成并校验后记账 · "+rateWindow+" 窗口":"Worker 样本新鲜 · 吞吐差分基线未知"):"Worker 样本未知或已过期";
  const stages=learnerFresh?safeArray(workspace.training_pipeline):[];
  const activity=learnerFresh?workspace.activity||null:null;
  const insights=safeArray(workspace.insights);
  const events=safeArray(workspace.events);
  const bepa=workspace.bepa||{};
  const observedLogical=workspace.last_observed_logical_run||null;
  const liveLogicalRun=learnerFresh&&snapshot.learner&&snapshot.learner.run||null;
  const logicalRun=liveLogicalRun||(observedLogical&&observedLogical.run)||null;
  const lastObservedLogicalRun=Boolean(logicalRun&&!liveLogicalRun);
  const status=learnerFresh&&logicalRun&&logicalRun.status?logicalRun.status:(snapshot?"Learner 状态未知":pathStatus(run,workspace.selection));
  const sampling=learnerFresh&&snapshot.learner&&snapshot.learner.sampling||{};
  const rewards=learnerFresh&&snapshot.learner&&snapshot.learner.rewards||{};
  const versions=learnerFresh?snapshot.learner||{}:{};
  const budget=bepa.scope||{};
  const ratio=(a,b)=>typeof a==="number"&&Number.isFinite(a)&&a>=0&&typeof b==="number"&&Number.isFinite(b)&&b>=0?fmtInt(a)+" / "+fmtInt(b):DASH;
  const statusStrip=panel("并行任务与学习信号",grid([
    metric("活跃题组",ratio(sampling.active_groups,sampling.maximum_groups),"组",learnerFresh?"Learner fresh · 实时 / 最大并发题数":"Learner freshness 未确认"),
    metric("原生 exact",ratio(rewards.native_exact_episodes,rewards.native_scored_episodes),"轨迹","原生成功 / 已评分；不含 replay"),
    metric("原生有效信号",ratio(rewards.native_variable_groups,rewards.scored_groups),"组","奖励有差异 / 已评分题组"),
    metric("策略 / serve",ratio(versions.policy_version,versions.serve_policy_version),"版本","训练策略和已发布版本"),
    metric("replay 组记录",fmtInt(rewards.replay_groups),"组","有replay reservation记录的组；不是当前预留或提交次数"),
    metric("Teacher 可派额度",fmtInt(budget.available===true?budget.remaining_tokens:null),"tokens",budget.id?"扣除已用及预留 · "+budget.id:"当前 scope 未知")
  ],"six"),{eyebrow:"TASKS · SIGNAL · POLICY",meta:sourceNote("monitor rewards / BEPA selected scope")});
  const hero=el("section",{className:"overview-hero"},
    el("div",{className:"hero-main"},el("div",{className:"eyebrow"},"QWEN RL · TRAINING OPERATIONS"),
      el("h2",{},run.run_id||(workspace.selection&&workspace.selection.observation_only?"当前 Learner 容器":"尚未选择 run")),
      el("p",{},warning||"训练、采样与证据状态按各自数据来源显示。"),
      logicalRun?el("div",{className:"hero-activity"},el("span",{},"训练任务 "+logicalRun.run_id),badge(logicalRun.status||"unknown"),el("small",{},String(logicalRun.completed_updates??"—")+" / "+String(logicalRun.requested_steps??"—")+" updates"),lastObservedLogicalRun?badge("最后观测记录","warn"):null):null,
      activity&&activity.phase?el("div",{className:"hero-activity"},badge(activity.state||"phase"),el("span",{},"Learner "+activity.phase),activity.elapsed_seconds!==null&&activity.elapsed_seconds!==undefined?el("small",{},fmtDuration(activity.elapsed_seconds)+" · "+(activity.time_basis||"time basis unknown")):el("small",{},activity.time_basis||"phase elapsed unknown")):null,
      el("div",{className:"hero-tags"},badge(status),workspace.selection&&workspace.selection.live_selected?badge("LIVE","good"):badge(workspace.selection&&workspace.selection.view_mode==="history"?"历史 run":"即时未知","warn"),workspace.source&&workspace.source.kind?el("span",{className:"tag"},workspace.source.kind):null)),
    el("div",{className:"hero-side"},el("span",{className:"hero-label"},"最近记录"),el("strong",{},record?String(record.record_id||"—"):"—"),el("span",{},record&&record.status?badge(record.status):"暂无记录"),el("small",{},record&&record.finished_at?ago(record.finished_at):"时间未知")));
  const ratePanel=panel("吞吐与有效学习",grid([
    metric("已校验输出",fmt(outputRate),"tok/s",workerMetricMeta,"teal"),
    metric("逻辑 run 成功更新",fmtInt(updates),"updates",learnerFresh?"逻辑 run 累计；含恢复前已完成更新":(record&&!snapshot?"selected run record":"Learner freshness 未确认"),""),
    metric("Episode refs",workersFresh?workerLeaseCount(snapshot):DASH,"refs","Worker leases；独立于 GPU utilization",""),
  ],"three"),{eyebrow:snapshot&&watchFresh?"LIVE TELEMETRY":"RUN EVIDENCE",description:"Worker 输出在请求完成后记账；Executor FWD / BWD 只显示各自最近完成的 CUDA 微批，异步 executor 不合并。",meta:sourceNote(snapshot&&watchFresh?"watch.snapshot + per-executor CUDA events":"selected run records","teal")});
  const executorPanel=panel("Executor 前向 / 反向实测",executorRates(snapshot,learnerFresh),{eyebrow:"COMPLETED CUDA MICRO-BATCH",description:"每行是单个 executor 的最新已完成微批速率，附实测年龄和当前阶段。闲置时只标注同一任务留下的最近测量；不估算在途速率，也不汇总异步 executor。",meta:sourceNote("current control executor operation telemetry")});
  const chart=el("div",{className:"history-chart"});
  const historyPanel=panel("吞吐历史",chart,{eyebrow:"OBSERVATION WINDOW",description:"Native 解码包含在途输出；已校验输出在请求完成后记账。Executor FWD / BWD 使用各自最新完成的 CUDA 微批事件；不同 executor 不合并，缺测和身份变化处断线。",meta:sourceNote("最新样本 "+((workspace.history||[]).length?new Date(workspace.history[workspace.history.length-1].sampled_at*1000).toLocaleTimeString("zh-CN",{hour12:false}):"尚无")+" · "+(workspace.history||[]).length+" 点")});
  const pipelinePanel=panel("训练与采样流水",pipelineView(stages,workspace.selection&&workspace.selection.live_selected),{eyebrow:"TRAINING PIPELINE",description:"请求、题组、更新与交接使用各自单位；不换算成百分比或合成吞吐。",meta:activity&&activity.phase?sourceNote("Learner "+activity.phase+(activity.elapsed_seconds!==null&&activity.elapsed_seconds!==undefined?" · "+fmtDuration(activity.elapsed_seconds):"")):sourceNote("activity unknown")});
  const workerPanel=panel("Worker / 硬件概览",workerSummary(snapshot,watchFresh,workspace.source),{eyebrow:"CAPACITY",description:"请求槽位、episode lease 与物理 GPU 分开显示。"});
  const bepaPanel=panel("BEPA 任务时间线",el("div",{className:"drawer-stack"},bepaCoverage(bepa),timeline(bepa)),{eyebrow:"TEACHER-ASSISTED DATA",description:"同批 timeline stage 覆盖计数；plan 可复用，cache reservation 与 optimizer commit 分开记录。",meta:bepa.available?sourceNote(bepa.source||"BEPA ledger"):"台账不可用"});
  const issuePanel=panel("需要关注",issueList(insights,events),{eyebrow:"DIAGNOSTICS",description:"只呈现后端明确给出的诊断。" });
  const output=el("div",{className:"view-stack"},
    warning && workspace.selection.view_mode!=="history" ?el("div",{className:"notice notice-"+(workspace.selection&&workspace.selection.live_selected===false?"info":"warn")},el("span",{className:"notice-mark"},workspace.selection&&workspace.selection.live_selected===false?"i":"!"),el("div",{},el("b",{},workspace.selection&&workspace.selection.live_selected===false?"历史复盘模式":"实时状态提示"),el("p",{},warning))):null,
    hero,ratePanel,executorPanel,statusStrip,renderEvaluationOverview(workspace.evaluation),grid([historyPanel,pipelinePanel],"wide-left"),workerPanel,grid([bepaPanel,issuePanel],"equal"));
  return {element:output,mount:()=>window.echarts&&chart.isConnected?mountHistoryChart(chart,workspace.history||[],document.documentElement.dataset.theme==="dark"):null};
}
function pathStatus(run,selection){return (run.manifest&&run.manifest.status)||(selection&&selection.observation_only?"observing":selection&&selection.live_selected?"running":null)||"状态未知";}
function pathUpdate(runCommit){const value=runCommit.logical_completed_updates;return typeof value==="number"&&Number.isFinite(value)&&value>=0?value:null;}
function workerLeaseCount(snapshot){const workers=safeArray(snapshot.workers);return workers.length&&workers.every((worker)=>worker.live===true&&typeof worker.episode_refs==="number"&&Number.isFinite(worker.episode_refs)&&worker.episode_refs>=0)?fmtInt(workers.reduce((total,worker)=>total+worker.episode_refs,0)):DASH;}
