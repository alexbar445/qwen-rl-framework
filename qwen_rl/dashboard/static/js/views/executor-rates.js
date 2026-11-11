import {DASH,el,emptyState,fmt,fmtDuration,fmtInt,safeArray} from "../core.js";

function executorPhaseLabel(row){
  const phase=row.current_phase&&row.current_phase.name;
  const index=row.current_phase&&row.current_phase.microbatch_index;
  if(row.status==="stale")return "Executor 采样过期"+(row.reported_phase&&row.reported_phase.name?" · 上次观测 "+row.reported_phase.name.toUpperCase():"");
  if(row.status==="idle")return "闲置";
  if(row.status==="waiting")return "任务已取 · 微批阶段未知";
  if(row.status==="active"&&phase)return "当前 "+phase.toUpperCase()+" 微批"+(Number.isInteger(index)?" #"+index:"")+(typeof row.phase_elapsed_seconds==="number"?" · "+fmtDuration(row.phase_elapsed_seconds):"");
  return row.reason||"当前阶段未知";
}
function executorRateCell(row,phase){
  const measurement=row.measurements&&row.measurements[phase];
  const value=measurement&&typeof measurement.tokens_per_second==="number"&&Number.isFinite(measurement.tokens_per_second)?fmt(measurement.tokens_per_second):DASH;
  const age=measurement&&typeof measurement.age_seconds==="number"?" · 距完成 "+fmtDuration(measurement.age_seconds):" · 完成时间未知";
  const label=!measurement?"尚无已完成 CUDA 微批":measurement.status==="stale"||row.status==="stale"?"采样过期 · 最近已完成":measurement.status==="last_completed_idle"?"闲置 · 最近已完成":row.current_phase&&row.current_phase.name===phase?"当前阶段 · 最近已完成":"最近已完成";
  const completed=measurement&&typeof measurement.completed_at_unix_seconds==="number"?new Date(measurement.completed_at_unix_seconds*1000).toLocaleTimeString("zh-CN",{hour12:false}):null;
  const detail=measurement?"微批 #"+measurement.microbatch_index+" · "+fmtInt(measurement.input_tokens)+" input tokens · CUDA "+fmtDuration(measurement.cuda_elapsed_seconds):null;
  return el("div",{className:"executor-rate"},el("b",{},value+(value===DASH?"":" tok/s")),el("small",{},label+age),detail?el("small",{},detail):null,completed?el("small",{},"实测 "+completed):null);
}
export function executorRates(snapshot,fresh){
  if(!fresh)return emptyState("Executor 前向 / 反向未知","Learner watch 不新鲜；不沿用可能过期的执行器相位或速率。");
  const telemetry=snapshot&&snapshot.execution_telemetry||{};
  const rows=safeArray(telemetry.executors);
  if(!rows.length)return emptyState("Executor 遥测未知",telemetry.reason||"当前控制配置没有可验证的 executor 微批遥测。");
  return el("div",{className:"table-scroll",dataset:{scrollKey:"executor-rates"}},el("table",{className:"data-table"},
    el("thead",{},el("tr",{},el("th",{},"Executor"),el("th",{},"当前阶段"),el("th",{},"FWD 单微批"),el("th",{},"BWD 单微批"))),
    el("tbody",{},rows.map((row)=>el("tr",{},
      el("td",{},el("b",{},row.executor_id||"未知"),el("small",{},row.reason||"")),
      el("td",{},executorPhaseLabel(row)),
      el("td",{},executorRateCell(row,"forward")),
      el("td",{},executorRateCell(row,"backward")))))));
}
