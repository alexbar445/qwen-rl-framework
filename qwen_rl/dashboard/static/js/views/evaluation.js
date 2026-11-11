import {openDrawer,DASH,ago,badge,compact,dataTable,el,emptyState,fmtDate,fmtInt,fmtDuration,fmt,grid,jsonDetails,kvRows,metric,panel,safeArray,sourceNote} from "../core.js";


function exactSummary(summary){
  const exact=summary&&(summary.native_exact||summary.nativeExact)||{};
  const count=exact.count ?? (summary && summary.exact_solve_count) ?? null;
  const known=exact.total ?? (summary && summary.exact_known_count) ?? null;
  const total=summary&&(summary.task_count ?? summary.tasks_total ?? summary.total_tasks);
  return {count,known,total,complete:summary&&summary.complete_valid_evaluation===true,unknown:summary&&summary.unknown_exact_count};
}
function completion(result){
  const s=result.summary||{};
  const done=result.tasks_done ?? s.tasks_done ?? s.completed_tasks ?? null,total=result.task_count ?? s.task_count ?? s.tasks_total ?? s.total_tasks ?? null;
  if(done!==null&&total!==null)return fmtInt(done)+" / "+fmtInt(total);
  if(s.complete_valid_evaluation===true&&total!==null)return fmtInt(total)+" / "+fmtInt(total);
  return DASH;
}
function scoreText(summary){
  const item=exactSummary(summary);
  if(item.count===null||item.known===null)return DASH;
  return fmtInt(item.count)+" / "+fmtInt(item.known)+" known"+(item.complete?" · complete":"");
}
function detail(result){
  const summary=result.summary||{};
  const exact=exactSummary(summary);
  openDrawer((result.kind||"Evaluation")+" · policy "+(result.policy_version===null||result.policy_version===undefined?"?":result.policy_version),el("div",{className:"drawer-stack"},
    el("div",{className:"evidence-note evidence-info"},el("strong",{},"Native evaluation evidence"),el("p",{},"仅使用独立 native ARC evaluation 的 persisted summary。未完成题目的 exact outcome 仍保持未知。")),
    kvRows([["Request",result.request_id],["Kind",result.kind],["Status",result.status],["Policy version",result.policy_version],["Adapter SHA-256",result.adapter_sha256],["Successful update count",result.successful_update_count],["Milestone update count",result.milestone_update_count],["Tasks done",completion(result)],["Known exact",scoreText(summary)],["Unknown exact",exact.unknown],["Complete valid evaluation",exact.complete?"yes":"no"],["Result ref",result.result_ref],["SHA-256",result.result_sha256],["Time",fmtDate(result.completed_at||result.created_at||result.at)]]),
    result.comparison?jsonDetails("Comparison evidence",result.comparison):null,
    jsonDetails("Persisted result summary",summary,48000)), "NATIVE EVALUATION");
}
function comparisonRows(results,baseline){
  const rows=[];
  if(baseline&&!results.some((item)=>item.request_id===baseline.request_id))rows.push(Object.assign({row_key:baseline.request_id,policy_version:0},baseline,{kind:"baseline"}));
  for(const item of results)rows.push(Object.assign({row_key:item.request_id},item));
  return rows;
}
export function renderEvaluation(workspace){
  const evaluation=workspace.evaluation||{};
  if(!evaluation.available)return el("div",{className:"view-stack"},el("header",{className:"section-title"},el("div",{className:"eyebrow"},"INDEPENDENT ARC EVALUATION"),el("h2",{},"Evaluation"),el("p",{},"Frozen taskset、baseline、checkpoint result 与 native exact outcome。")),el("div",{className:"notice notice-info"},el("span",{className:"notice-mark"},"i"),el("div",{},el("b",{},"Evaluation evidence unavailable"),el("p",{},evaluation.reason||"当前 run 未提供独立评估数据。"))),evaluation.source?jsonDetails("Evaluation source",evaluation.source):null);
  const results=safeArray(evaluation.results),baseline=evaluation.baseline||null,latest=evaluation.latest||results[results.length-1]||baseline;
  const taskset=evaluation.evalset_sha256||latest&&latest.summary&&latest.summary.evalset_sha256||null;
  const rows=comparisonRows(results,baseline);
  const resultTable=dataTable("evaluation-results",rows,[
    {key:"kind",label:"Run"},
    {key:"status",label:"状态",render:(row)=>badge(row.status)},
    {key:"policy_version",label:"Policy",numeric:true},
    {key:"successful_update_count",label:"Successful updates",numeric:true},
    {key:"milestone_update_count",label:"Milestone",numeric:true},
    {key:"summary.native_exact.count",label:"Native exact",sortValue:(row)=>row.summary&&row.summary.native_exact&&row.summary.native_exact.count,render:(row)=>scoreText(row.summary||{})},
    {key:"summary.complete_valid_evaluation",label:"Completeness",render:(row)=>row.summary&&row.summary.complete_valid_evaluation===true?badge("complete","good"):badge("partial / unknown","warn")},
    {key:"summary.gains",label:"新增成功",numeric:true,render:(row)=>row.summary&&row.summary.vs_baseline&&row.summary.vs_baseline.complete_paired_evaluation===true?fmtInt(row.summary.gains):DASH},
    {key:"summary.regressions",label:"退步",numeric:true,render:(row)=>row.summary&&row.summary.vs_baseline&&row.summary.vs_baseline.complete_paired_evaluation===true?fmtInt(row.summary.regressions):DASH},
    {key:"summary.generated_tokens",label:"輸出 tokens",numeric:true,render:(row)=>fmtInt(row.summary&&row.summary.generated_tokens)},
    {key:"summary.duration_seconds_total",label:"Episode 累計時間",numeric:true,render:(row)=>row.summary&&typeof row.summary.duration_seconds_total==="number"?fmtDuration(row.summary.duration_seconds_total):DASH},
    {key:"summary.tool_error_rate",label:"工具錯誤率",numeric:true,render:(row)=>row.summary&&typeof row.summary.tool_error_rate==="number"?fmt(row.summary.tool_error_rate*100)+"%":DASH},
    {key:"adapter_sha256",label:"Adapter SHA",render:(row)=>row.adapter_sha256?el("span",{className:"mono"},compact(row.adapter_sha256,16)):DASH},
    {key:"completed_at",label:"Completed",sortValue:(row)=>row.completed_at||row.created_at||row.at,render:(row)=>row.completed_at||row.created_at||row.at?ago(row.completed_at||row.created_at||row.at):DASH},
  ],{pageSize:12,note:"打开行查看独立评估结果与配对证据",onRow:detail});
  const summaryCard=latest&&latest.summary?jsonDetails("Latest persisted summary",latest.summary,48000):emptyState("没有已完成结果","未将实时 policy 或训练 reward 作为 evaluation 分数。");
  const output=el("div",{className:"view-stack"},
    el("header",{className:"section-title"},el("div",{className:"eyebrow"},"INDEPENDENT ARC EVALUATION"),el("h2",{},"Evaluation"),el("p",{},"对 frozen taskset 的 baseline 与 policy checkpoint 结果；训练 reward、guided cache 与评估分数分开。")),
    evaluation.selected_lineage?el("div",{className:"lineage-banner"},el("span",{className:"eyebrow"},"SELECTED EVALUATION LINEAGE"),el("b",{},(typeof evaluation.selected_lineage==="string"?evaluation.selected_lineage:evaluation.selected_lineage.lineage_id)||evaluation.lineage_id||"lineage unknown"),el("span",{className:"mono"},(typeof evaluation.selected_lineage==="object"&&evaluation.selected_lineage.evalset_sha256)||taskset||"taskset SHA unknown")):null,
    grid([
      metric("Baseline exact",baseline?scoreText(baseline.summary||{}):DASH,"native tasks",baseline?(baseline.status||"baseline result"):"baseline result unavailable", "blue"),
      metric("Latest exact",latest?scoreText(latest.summary||{}):DASH,"native tasks",latest&&latest.kind?latest.kind:"no persisted result","teal"),
      metric("Task completion",latest?completion(latest):DASH,"tasks",latest&&latest.summary&&latest.summary.complete_valid_evaluation===true?"complete valid set":"partial coverage stays explicit","amber"),
      metric("Checkpoint results",fmtInt(results.length),"records","only persisted evaluation requests",""),
    ],"four"),
    taskset?panel("Frozen evaluation taskset",kvRows([["Evalset SHA-256",el("span",{className:"mono"},taskset)],["Lineage",evaluation.lineage_id||evaluation.selected_lineage&&evaluation.selected_lineage.lineage_id],["Successful updates",fmtInt(evaluation.lineage_successful_update_count)],["Baseline request",baseline&&baseline.request_id],["Baseline status",baseline&&baseline.status]]),{eyebrow:"DATASET PROVENANCE",meta:sourceNote(evaluation.source||"evaluation state")}):panel("Frozen evaluation taskset",emptyState("Taskset identity unavailable","No hash is inferred from a directory name."),{eyebrow:"DATASET PROVENANCE"}),
    panel("Baseline ↔ checkpoint results",resultTable,{eyebrow:"NATIVE ONLY",description:"exact 的分子与已知分母显示原始统计；未完成 / unknown tasks 不作为失败。",flush:true}),
    panel("Latest result details",summaryCard,{eyebrow:"COMPARISON + TASK OUTCOMES"}),
    evaluation.latest&&evaluation.latest.comparison?grid([panel("Baseline comparison",jsonDetails("comparison.baseline",evaluation.latest.comparison.baseline_pairing||evaluation.latest.comparison.baseline||{}),{eyebrow:"PAIRED EVIDENCE"}),panel("Previous checkpoint",jsonDetails("comparison.previous",evaluation.latest.comparison.previous_pairing||evaluation.latest.comparison.previous||{}),{eyebrow:"POLICY PROGRESSION"})],"two"):null
  );
  return output;
}


export function renderEvaluationOverview(evaluation={}){
  if(!evaluation.available)return panel("獨立評估",el("div",{className:"quiet-callout"},evaluation.reason||"評估證據尚未就緒"));
  const rows=safeArray(evaluation.results);
  const baseline=evaluation.baseline;
  const complete=rows.filter((row)=>row.kind!=="baseline"&&row.summary&&row.summary.complete_valid_evaluation===true);
  const latest=complete[complete.length-1];
  const score=(row)=>row&&row.summary&&row.summary.complete_valid_evaluation===true?scoreText(row.summary):"待完整結果";
  const count=evaluation.lineage_successful_update_count,interval=evaluation.milestone_interval;
  const next=Number.isInteger(count)&&Number.isInteger(interval)&&interval>0?(Math.floor(count/interval)+1)*interval:null;
  return panel("獨立評估 · 無 Teacher / replay",el("div",{className:"drawer-stack"},
    grid([metric("v0 原生 exact",score(baseline),"", "同底座、零 LoRA 殘差"),
          metric(latest?"v"+latest.policy_version+" 原生 exact":"RL checkpoint 原生 exact",score(latest),"","完整固定題集"),
          metric("成功 RL 更新",fmtInt(count),"次",next===null?"下一次評估未知":"下一次評估：第 "+next+" 次")],"three"),
    latest?kvRows([["相對 v0 新增／退步",latest.summary.vs_baseline&&latest.summary.vs_baseline.complete_paired_evaluation===true?fmtInt(latest.summary.gains)+" / "+fmtInt(latest.summary.regressions)+" 題":DASH],["本次輸出 tokens",fmtInt(latest.summary.generated_tokens)]]):null,
    ...rows.filter((row)=>row.status!=="completed").map((row)=>el("div",{className:"runtime-strip"},badge(row.status),el("b",{},"v"+row.policy_version),el("span",{},completion(row)+" 題完成"))),
    el("a",{href:"#/evaluation"},"查看逐次結果、token／耗時與新增成功／退步題 →")
  ),{eyebrow:"FIXED HELD-OUT · NATIVE EXACT",description:"只顯示完整評估分數；未完成與執行錯誤保持未知。"});
}
