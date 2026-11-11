// Interpret the current watch activity/compute contracts without synthesizing samples.
const LABELS={inference_inflight:"生成请求在途",queued:"等待调度",between_generations:"episode 引用保留",not_ready:"未就绪 / 载入中",unknown:"未知",idle:"闲置"};
function activityLabel(state){
  if(LABELS[state])return LABELS[state];
  if(state.startsWith("wait_dispatch:"))return "派送等待："+state.slice(14);
  if(state.startsWith("compute:"))return "计算："+state.slice(8);
  if(state.startsWith("foreground:"))return "前台："+state.slice(11);
  return state;
}
export function activityEvidence(activity){
  const states=activity&&activity.states;
  if(!states||typeof states!=="object"||Array.isArray(states))return [];
  return Object.entries(states).map(([state,item])=>{
    const value=item&&item.percent_of_observed_worker_walltime;
    const percent=typeof value==="number"&&Number.isFinite(value)&&value>=0&&value<=100?value:null;
    return {state,label:activityLabel(state),percent};
  });
}
export function learnerTopology(compute){
  const topology=compute&&compute.topology||{};
  const reported=compute&&compute.world_size;
  const worldSize=Number.isInteger(reported)&&reported>0?reported:
    Number.isInteger(topology.world_size)&&topology.world_size>0?topology.world_size:null;
  const devices=Array.isArray(compute&&compute.devices)?compute.devices:
    Array.isArray(topology.devices)?topology.devices:[];
  return {mode:compute&&compute.mode||topology.mode||null,worldSize,devices};
}
