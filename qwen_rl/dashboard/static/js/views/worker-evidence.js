// Runtime identity and host telemetry are separate, explicitly scoped evidence.
export function workerEvidence(worker,source,gpus=[],snapshot=null){
  const runtime=worker.runtime||{};
  let loopback=false;
  try{loopback=["127.0.0.1","localhost","[::1]"].includes(new URL(worker.endpoint).hostname);}catch{}
  const local=worker.scope==="local"||worker.local===true||(
    source&&source.kind==="local"&&worker.scope==="service"&&
    worker.site_id==="central"&&worker.node_id==="central"&&loopback);
  const gpu=local&&Number.isInteger(worker.gpu)?gpus.find((row)=>row.index===worker.gpu)||null:null;
  const phase=snapshot&&snapshot.learner&&snapshot.learner.gpu_phases||{};
  const admissions=snapshot&&snapshot.learner&&snapshot.learner.sampling&&snapshot.learner.sampling.worker_resource_admission||{};
  const learnerLive=Boolean(snapshot&&snapshot.freshness&&snapshot.freshness.learner&&snapshot.freshness.learner.live===true);
  const shared=phase.mode==="phase_shared"&&Array.isArray(phase.shared_worker_ids)&&phase.shared_worker_ids.includes(worker.id);
  const phaseLabels={draining_workers:"正在结束在途任务，准备让卡",training:"已让卡给 Learner",parking_learner:"等待 Learner 释放 GPU",checkpointing:"保存 checkpoint／并行唤醒 Worker，尚未开放采样",resident_fenced:"恢复 Worker／准备策略，尚未开放采样"};
  const sharedActivity=learnerLive&&shared&&admissions[worker.id]==="phase_paused"?phaseLabels[phase.state]:null;
  return {runtime,local,gpu,location:(local?"本机 GPU ":"节点 GPU ")+(Number.isInteger(worker.gpu)?worker.gpu:"未知"),
    activity:worker.live===true?(sharedActivity||worker.activity_state||"状态未报告"):"观测过期"};
}
