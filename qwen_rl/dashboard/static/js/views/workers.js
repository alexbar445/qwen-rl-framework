import {DASH,badge,compact,el,emptyState,fmt,fmtBytes,fmtDuration,fmtInt,fmtPct,grid,jsonDetails,kvRows,metric,panel,safeArray,sourceNote} from "../core.js";
import {liveSnapshot} from "./common.js";
import {workerEvidence} from "./worker-evidence.js";
import {activityEvidence,learnerTopology} from "./workers-data.js";

function groupWorkers(rows){
  const groups=new Map();
  for(const worker of rows){
    const site=worker.site_id||worker.scope||"node unknown",node=worker.node_id||worker.id||"worker";
    const key=String(site)+" / "+String(node);
    if(!groups.has(key))groups.set(key,[]);
    groups.get(key).push(worker);
  }
  return Array.from(groups,([label,workers])=>({label,workers}));
}
function workerCard(worker,source,gpus,snapshot){
  const kv=worker.kv_cache||{},cap=worker.capabilities||{};
  const evidence=workerEvidence(worker,source,gpus,snapshot),runtime=evidence.runtime;
  const kvValue=kv.usage_percent!==null&&kv.usage_percent!==undefined?fmtPct(kv.usage_percent,1):"未知 / 过期";
  const gpu=evidence.location;
  const residents=safeArray(worker.resident_versions);
  return el("article",{className:"worker-card"},
    el("div",{className:"worker-head"},el("div",{},el("div",{className:"worker-id"},worker.id||"Worker"),el("div",{className:"worker-location"},runtime.backend||worker.node_id||worker.scope||"节点未知")),badge(worker.live===true?"live":worker.live===false?"stale":"unknown")),
    el("div",{className:"worker-slot"},el("div",{},el("span",{},"Active generations"),el("b",{},worker.active===null||worker.active===undefined?DASH:(fmtInt(worker.active)+" / "+fmtInt(worker.maximum)))),worker.queued!==null&&worker.queued!==undefined?el("span",{className:"worker-queued"},fmtInt(worker.queued)+" queued"):null),
    el("div",{className:"worker-facts"},
      fact("资源位置",gpu),fact("当前状态",evidence.activity),fact("生成 tok/s",worker.live===true?fmt(worker.tokens_per_second):DASH),fact("GPU 计算",evidence.gpu?fmtPct(evidence.gpu.util_percent):DASH),fact("槽位使用",fmtPct(worker.slot_use_percent,0)),fact("KV 使用",kvValue),fact("KV 样本年龄",kv.age_seconds===undefined?"未知":fmtDuration(kv.age_seconds)),fact("Episode refs",worker.episode_refs===undefined?DASH:fmtInt(worker.episode_refs)),fact("Policy",worker.policy_version===undefined?DASH:"v"+worker.policy_version)),
    el("div",{className:"worker-identity"},
      identity("量化",runtime.quantization_mode||"未报告"),
      identity("4-bit loaded",runtime.is_loaded_in_4bit===undefined?"未知":(runtime.is_loaded_in_4bit?"是":"否")),
      identity("Adapter SHA",runtime.adapter_hash||"未知")),
    residents.length?el("div",{className:"resident-list"},el("span",{className:"eyebrow"},"RESIDENT ADAPTERS"),...residents.map((item)=>el("span",{className:"resident-chip"},"v",item.policy_version," · ",compact(item.adapter_hash||"",12)," · ep ",item.episode_refs," / gen ",item.generation_refs))):null,
    el("details",{className:"capability-details"},el("summary",{},"模型与生成能力"),jsonDetails("Worker capability",Object.assign({},cap,{
      backend:runtime.backend,quantization_mode:runtime.quantization_mode,is_loaded_in_4bit:runtime.is_loaded_in_4bit,
      artifact_fingerprint:runtime.artifact_fingerprint,base_fingerprint:runtime.base_fingerprint,adapter_hash:runtime.adapter_hash,
      max_model_len:runtime.max_model_len,max_generated_tokens:runtime.max_generated_tokens,
      speculative_decoding:runtime.speculative_decoding,prefix_caching_enabled:runtime.prefix_caching_enabled,
      packed_module_count:runtime.packed_module_count
    }))));
}
function fact(label,value){return el("div",{className:"worker-fact"},el("span",{},label),el("b",{},value===null||value===undefined?DASH:String(value)));}
function identity(label,value){return el("div",{className:"identity-row"},el("span",{},label),el("b",{className:label.includes("SHA")?"mono":""},value));}
function localGpu(snapshot){
  const cards=safeArray(snapshot.gpus);
  if(!cards.length)return emptyState("本机 GPU 读数未知","nvidia-smi 无有效结果；不推断 GPU 使用率为 0。");
  return el("div",{className:"gpu-grid"},cards.map((gpu)=>el("article",{className:"gpu-card"},
    el("div",{className:"gpu-head"},el("b",{},"GPU "+gpu.index),el("span",{},compact(gpu.name||"型号未知",34))),
    el("div",{className:"gpu-bar-row"},el("span",{},"计算"),el("div",{className:"meter "+(typeof gpu.util_percent==="number"?"":"meter-unknown")},typeof gpu.util_percent==="number"?el("i",{style:{width:Math.max(0,Math.min(100,gpu.util_percent))+"%"}}):null),el("b",{},fmtPct(gpu.util_percent))),
    el("div",{className:"gpu-bar-row"},el("span",{},"显存"),el("div",{className:"meter meter-warm "+(typeof gpu.memory_used_mib==="number"&&typeof gpu.memory_total_mib==="number"&&gpu.memory_total_mib>0?"":"meter-unknown")},typeof gpu.memory_used_mib==="number"&&typeof gpu.memory_total_mib==="number"&&gpu.memory_total_mib>0?el("i",{style:{width:Math.min(100,100*gpu.memory_used_mib/gpu.memory_total_mib)+"%"}}):null),el("b",{},typeof gpu.memory_used_mib==="number"&&typeof gpu.memory_total_mib==="number"?fmtInt(gpu.memory_used_mib)+" / "+fmtInt(gpu.memory_total_mib)+" MiB":DASH)),
    el("div",{className:"gpu-caption"},fmt(gpu.power_watts,0)+" W · "+fmt(gpu.temperature_c,0)+" °C · "+fmt(gpu.memory_clock_mhz,0)+" MHz"))));
}
function activityView(activity){
  if(!activity)return emptyState("活动窗口未知","watch 未提供状态取样。");
  const states=activityEvidence(activity);
  if(!states.length)return emptyState("活动状态未知","当前采样窗口没有已报告的状态比例。");
  return el("div",{className:"activity-list"},states.map((item)=>el("div",{className:"activity-row"},el("span",{},item.label),el("div",{className:"meter "+(item.percent===null?"meter-unknown":"")},item.percent===null?null:el("i",{style:{width:item.percent+"%"}})),el("b",{},fmtPct(item.percent,1)))),
    el("div",{className:"activity-meta"},fmtDuration(activity.observed_wall_seconds)+" observed / "+fmtDuration(activity.expected_wall_seconds)+" expected · known "+fmtPct(activity.known_percent,0)+" · unknown "+fmtPct(activity.unknown_percent,0)),
    el("div",{className:"activity-meta"},"比例按已观测 wall time 计算；不是 GPU 利用率。历史 not_ready 样本无法反推为让卡状态。"));
}
export function renderWorkers(workspace){
  const snapshot=liveSnapshot(workspace);
  if(!snapshot)return el("div",{className:"view-stack"},el("header",{className:"section-title"},el("div",{className:"eyebrow"},"CLUSTER"),el("h2",{},"Workers & cluster"),el("p",{},workspace.selection&&workspace.selection.live_selected===false?"历史 run 复盘会隐藏其他 run 的实时 Worker 与 GPU 读数。":(workspace.watch&&workspace.watch.error)||"当前 watch 快照不可用。")),el("div",{className:"notice notice-info"},el("span",{className:"notice-mark"},"i"),el("div",{},el("b",{},"实时集群状态未知"),el("p",{},"不会用历史或旧 Worker 数据填充当前集群状态。"))));
  const rows=safeArray(snapshot.workers),groups=groupWorkers(rows),learner=snapshot.learner||{},health=snapshot.learner_health||{},compute=health.compute||{},topology=learnerTopology(compute);
  const parallelism=topology.mode||"未知";
  const cards=groups.length?groups.map((group)=>el("section",{className:"node-group"},el("div",{className:"node-heading"},el("div",{},el("div",{className:"eyebrow"},"NODE / SITE"),el("h3",{},group.label)),el("span",{className:"node-count"},group.workers.length+" workers")),el("div",{className:"worker-grid"},group.workers.map((worker)=>workerCard(worker,workspace.source,safeArray(snapshot.gpus),snapshot))))):emptyState("没有 Worker 行","watch 快照未包含 Worker 列表。");
  const healthRows=health&&Object.keys(health).length?jsonDetails("Learner health evidence",health):emptyState("Learner health 未提供","不从配置或历史数据反推当前进程的运行能力。");
  const parallel=panel("Learner 并行布局",el("div",{className:"parallel-layout"},
    metric("运行模式",parallelism,"",topology.worldSize!==null?"world size "+topology.worldSize:"world size 未报告","teal"),
    metric("Learner GPU ranks",topology.devices.length?fmtInt(topology.devices.length):DASH,"devices",topology.devices.length?"实际 compute 所报设备":"实际 compute 设备未知","blue"),
    metric("前向计量范围",learner.training&&learner.training.forward&&learner.training.forward.measurement_scope||DASH,"",learner.training&&learner.training.forward&&learner.training.forward.measurement_scope==="rank0_partial"?"DP rank0 partial; 不是全局吞吐":"按 watch scope 解读","amber")), {eyebrow:"RANK SCOPE",description:"TP 同一批次只计一次；DP 实时前向若标为 rank0_partial，则显示局部读数。"});
  return el("div",{className:"view-stack"},
    el("header",{className:"section-title"},el("div",{className:"eyebrow"},"CLUSTER"),el("h2",{},"Workers & cluster"),el("p",{},"按 remote site 与 node 分组；Worker lease、请求槽位、KV 与物理 GPU 保持独立口径。")),
    grid([metric("Worker 数",fmtInt(rows.length),"workers","当前 watch 快照中有记录的 Worker"),metric("活动 generation",sumKnown(rows,"active"),"requests","占槽率不等于 GPU 利用率"),metric("Episode refs",sumKnown(rows,"episode_refs"),"leases","已知引用求和；缺项显示未知")],"three"),
    panel("Worker 节点",cards,{eyebrow:"REMOTE / LOCAL INVENTORY",description:"远程 GPU 利用率只在对应节点提供观测时显示。"}),
    grid([panel("本机物理 GPU",localGpu(snapshot),{eyebrow:"NVIDIA-SMI",description:"物理卡包含同卡其他程序占用，不能当作本训练进程独占计量。"}),panel("Worker 活动窗口",activityView(snapshot.worker_activity),{eyebrow:"5 MINUTE ACTIVITY",description:"Worker inference 在途不等同 GPU 实际计算。"})],"two"),
    grid([panel("Learner 活动窗口",activityView(snapshot.learner_activity),{eyebrow:"LEARNER STATE"}),panel("Learner health",healthRows,{eyebrow:"IDENTITY + COMPUTE",description:"包含量化、rank evidence、最大长度与恢复状态。"})],"two"),
    parallel,
    panel("完整 Worker rows",jsonDetails("Worker rows",rows),{eyebrow:"RAW WATCH EVIDENCE"})
  );
}
function sumKnown(rows,key){const values=rows.map((row)=>row[key]);return values.length&&values.every((value)=>typeof value==="number")?fmtInt(values.reduce((a,b)=>a+b,0)):DASH;}
