import {DASH,ago,badge,dataTable,el,emptyState,fmtBytes,fmtDate,fmtInt,grid,jsonDetails,kvRows,metric,panel,safeArray,makeLink,openDrawer} from "../core.js";
import {groups,openTrajectory,records} from "./common.js";

const NOT_RECORDED="未记录";
function present(value){return value!==null&&value!==undefined&&value!=="";}
function label(value){return present(value)?String(value):NOT_RECORDED;}
function vectorLabel(value){
  if(!Array.isArray(value))return NOT_RECORDED;
  return value.length?value.map((item)=>present(item)?String(item):NOT_RECORDED).join(" · "):"已记录（空向量）";
}
function exactCount(group){
  const value=group.exact;
  if(typeof value==="boolean")return {count:value?1:0,total:1,complete:true};
  if(!Array.isArray(value))return null;
  const known=value.filter((item)=>typeof item==="boolean");
  if(!known.length)return null;
  return {count:known.filter(Boolean).length,total:known.length,complete:known.length===value.length};
}
function exactLabel(group){
  const result=exactCount(group);
  if(result)return result.count+" / "+result.total+(result.complete?"":"（已知）");
  if(Array.isArray(group.exact))return NOT_RECORDED+"（"+group.exact.length+" 行）";
  return NOT_RECORDED;
}
function optimizerEvidence(group){
  if(group.optimizer_consumed===true)return {label:"已消费",tone:"good",basis:"group.optimizer_consumed=true"};
  if(group.optimizer_consumed===false)return {label:"未消费",tone:"warn",basis:"group.optimizer_consumed=false"};
  if(group.status==="filtered_zero_variance")return {label:"未提交（过滤）",tone:"neutral",basis:"group.status=filtered_zero_variance；optimizer_consumed 未记录"};
  return {label:NOT_RECORDED,tone:"unknown",basis:"group.optimizer_consumed 未记录"};
}
function submissionField(group,key){
  if(present(group[key]))return String(group[key]);
  if(group.optimizer_consumed===false)return "不适用（optimizer_consumed=false，组未被消费）";
  if(group.status==="filtered_zero_variance")return "不适用（status=filtered_zero_variance，组已过滤）";
  return NOT_RECORDED;
}
function cardinalitiesLabel(value){
  if(!value||typeof value!=="object"||Array.isArray(value))return NOT_RECORDED;
  const entries=Object.entries(value);
  return entries.length?entries.map(([key,item])=>key+": "+label(item)).join(" · "):"已记录（空对象）";
}
function preparedRowsLabel(group){
  const direct=present(group.training_group_size)?group.training_group_size:null;
  const cardinality=group.group_cardinalities&&present(group.group_cardinalities.training)?group.group_cardinalities.training:null;
  if(direct!==null&&cardinality!==null&&Number(direct)!==Number(cardinality))return label(direct)+"（cardinality: "+label(cardinality)+"）";
  return label(direct!==null?direct:cardinality);
}
function phaseLabel(group){
  if(present(group.phase))return String(group.phase);
  return present(group.trajectory_artifact)?"未记录，见轨迹":NOT_RECORDED;
}
function boolLabel(value){return typeof value==="boolean"?(value?"是":"否"):NOT_RECORDED;}
function sourceClass(group){
  if(Array.isArray(group.row_kinds)){
    const replay=group.row_kinds.some((kind)=>kind==="guided_replay"||kind==="native_replay");
    const fresh=group.row_kinds.some((kind)=>["native","fresh","fresh_student"].includes(kind));
    return replay?(fresh?"native_plus_replay":"replay_only"):fresh?"native_only":NOT_RECORDED;
  }
  if(group.selected_ready_cache===true||group.replay_reservation)return "native_plus_replay";
  return group.selected_ready_cache===false?"native_only":NOT_RECORDED;
}
export function groupEvidenceView(group={}){
  const trainingRewards=Array.isArray(group.training_rewards)
    ?vectorLabel(group.training_rewards)
    :group.record_status==="failed"?"未记录（record failed；准备行数不是已提交行数）":NOT_RECORDED;
  const observedTrainingRows=Array.isArray(group.training_rewards)
    ?String(group.training_rewards.length)
    :group.record_status==="failed"?"未记录（record failed）":NOT_RECORDED;
  return {
    taskId:label(group.task_id),groupId:label(group.group_id),phase:phaseLabel(group),status:label(group.status),recordStatus:label(group.record_status),
    exact:exactLabel(group),exactVector:typeof group.exact==="boolean"?String(group.exact):vectorLabel(group.exact),
    rewards:vectorLabel(group.rewards),sourceClass:sourceClass(group),rawGroupSize:label(group.raw_group_size??group.group_size),preparedTrainingSize:preparedRowsLabel(group),
    trainingRewards,observedTrainingRows,groupCardinalities:cardinalitiesLabel(group.group_cardinalities),
    selectedReadyCache:boolLabel(group.selected_ready_cache),optimizer:optimizerEvidence(group),
    learnerStart:submissionField(group,"learner_start_version"),policyLag:submissionField(group,"policy_lag"),
    replayRowsAdded:label(group.replay_rows_added_observed),trajectory:label(group.trajectory_artifact),
  };
}
export function failureEvidence(record={}){
  const error=record.error;
  const status=String(record.status||"").toLowerCase();
  if(status!=="failed"&&status!=="error"&&!present(error))return null;
  const details=error&&typeof error==="object"?error:{};
  const type=label(details.type);
  const message=typeof error==="string"?error:label(details.message);
  const oom=/outofmemory|out of memory|cuda.{0,40}memory/i.test(type+" "+message);
  return {
    label:oom?"CUDA OOM":"训练记录失败",recordId:label(record.record_id),status:label(record.status),
    type,message,runStage:label(details.run_stage),learnerStage:label(details.learner_stage),updates:label(record.updates),
  };
}
function groupDrawer(group){
  const evidence=groupEvidenceView(group);
  openDrawer("题组 "+(group.group_id||group.task_id||NOT_RECORDED),el("div",{className:"drawer-stack"},
    kvRows([
      ["Task ID",evidence.taskId],["Group",evidence.groupId],["记录",group.record_id],["记录状态",evidence.recordStatus],
      ["组状态",evidence.status],["Phase",evidence.phase],["Cohort source class",evidence.sourceClass],
      ["Policy lag",evidence.policyLag],["Learner start",evidence.learnerStart],
      ["Optimizer",evidence.optimizer.label+" · "+evidence.optimizer.basis],
      ["Raw group size",evidence.rawGroupSize],["Prepared training rows",evidence.preparedTrainingSize],
      ["group_cardinalities",evidence.groupCardinalities],["Observed training rows",evidence.observedTrainingRows],
      ["Replay rows observed",evidence.replayRowsAdded],["selected_ready_cache",evidence.selectedReadyCache],
    ]),
    panel("候选与训练向量",kvRows([
      ["Native exact · group.exact",evidence.exact+" · "+evidence.exactVector],
      ["Native rewards · group.rewards",evidence.rewards],
      ["training_rewards · 成功更新后记录",evidence.trainingRewards],
    ]),{eyebrow:"GROUP OUTCOME"}),
    jsonDetails("完整 canonical group evidence",group)),"GROUP / TRAJECTORY");
}
function groupTableRows(items){return items.map((group,index)=>Object.assign({row_index:index},group));}
function coverage(known,total){
  return Number.isFinite(known)&&Number.isFinite(total)?fmtInt(known)+" / "+fmtInt(total):NOT_RECORDED;
}
function cohortPhase(row){
  if(present(row.phase))return String(row.phase);
  return present(row.phase_basis)?"未记录，见轨迹 · "+String(row.phase_basis):NOT_RECORDED;
}
function cohortTable(rows){
  if(!rows.length)return emptyState("No persisted cohorts","Cohorts are aggregated by policy version, source class and phase.");
  return dataTable("group-cohorts",rows,[
    {key:"policy_version",label:"Policy",numeric:true},
    {key:"source_class",label:"Source class",render:(row)=>badge(label(row.source_class))},
    {key:"phase",label:"Phase",render:cohortPhase},
    {key:"groups",label:"Groups",numeric:true},
    {key:"native_candidate_rows",label:"Native rows",numeric:true,render:(row)=>present(row.native_candidate_rows)?fmtInt(row.native_candidate_rows):NOT_RECORDED},
    {key:"native_candidate_row_groups",label:"Native vectors",numeric:true,render:(row)=>coverage(row.native_candidate_row_groups,row.groups)},
    {key:"training_rows_observed",label:"Observed train rows",numeric:true,render:(row)=>present(row.training_rows_observed)?fmtInt(row.training_rows_observed):NOT_RECORDED},
    {key:"training_row_groups_known",label:"Observed vectors",numeric:true,render:(row)=>coverage(row.training_row_groups_known,row.groups)},
    {key:"prepared_rows",label:"Prepared rows",numeric:true,render:(row)=>present(row.prepared_rows)?fmtInt(row.prepared_rows):NOT_RECORDED},
    {key:"prepared_replay_rows",label:"Prepared replay",numeric:true,render:(row)=>present(row.prepared_replay_rows)?fmtInt(row.prepared_replay_rows):NOT_RECORDED},
    {key:"prepared_row_groups_known",label:"Prepared coverage",numeric:true,render:(row)=>coverage(row.prepared_row_groups_known,row.groups)},
    {key:"replay_rows_added_observed",label:"Observed replay rows",numeric:true,render:(row)=>present(row.replay_rows_added_observed)?fmtInt(row.replay_rows_added_observed):NOT_RECORDED},
    {key:"replay_row_groups_known",label:"Observed replay vectors",numeric:true,render:(row)=>coverage(row.replay_row_groups_known,row.groups)},
    {key:"native_exact_success",label:"Native exact",numeric:true,render:(row)=>row.native_exact_known>0?fmtInt(row.native_exact_success)+" / "+fmtInt(row.native_exact_known):NOT_RECORDED},
  ],{pageSize:12,note:"Prepared rows come from group cardinalities and do not prove optimizer consumption; observed rows require training_rewards",onRow:(row)=>openDrawer("Cohort · policy "+label(row.policy_version),el("div",{className:"drawer-stack"},kvRows([
    ["Source class",label(row.source_class)],["Phase",cohortPhase(row)],["Phase basis",label(row.phase_basis)],["Groups",row.groups],
    ["Prepared rows",label(row.prepared_rows)],["Prepared replay rows",label(row.prepared_replay_rows)],["Prepared group coverage",coverage(row.prepared_row_groups_known,row.groups)],
    ["Observed training rows",label(row.training_rows_observed)],["Native exact basis",label(row.exact_basis)],["Cohort basis",label(row.basis)],
    ["Run records",safeArray(row.record_ids).join(" · ")||NOT_RECORDED],
  ]),jsonDetails("Persisted cohort evidence",row)),"GROUP COHORT")});
}
function failuresPanel(failures){
  if(!failures.length)return null;
  return panel("Failed training records",el("div",{className:"guidance-list"},failures.map((item)=>el("div",{className:"evidence-note evidence-warn"},
    el("strong",{},item.label+" · "+item.recordId),
    kvRows([["Record status",item.status],["Error type",item.type],["Updates",item.updates],["Run stage",item.runStage],["Learner stage",item.learnerStage],["Error",item.message]])))),
    {eyebrow:"RUN FAILURE EVIDENCE",description:"失败记录与组状态分开显示；failed run 的 prepared rows 不表示已提交训练行。"});
}
function openGroup(row){groupDrawer(row);}
export function renderGroups(workspace){
  const groupRows=groupTableRows(groups(workspace));
  const files=safeArray(workspace.run&&workspace.run.trajectories);
  const allRecords=records(workspace);
  const exactTotals=groupRows.map((group)=>exactCount(group)).filter((item)=>item&&item.total>0);
  const cohortRows=safeArray(workspace.bepa&&workspace.bepa.cohorts);
  const failedRecords=allRecords.map((record)=>failureEvidence(record)).filter(Boolean);
  const exactCountTotal=exactTotals.length?exactTotals.reduce((sum,item)=>sum+item.count,0):null;
  const exactKnownTotal=exactTotals.length?exactTotals.reduce((sum,item)=>sum+item.total,0):null;
  const consumedKnown=groupRows.filter((group)=>typeof group.optimizer_consumed==="boolean");
  const consumed=consumedKnown.length?fmtInt(consumedKnown.filter((group)=>group.optimizer_consumed===true).length)+" / "+fmtInt(consumedKnown.length)+" known":NOT_RECORDED;
  const updateValues=allRecords.map((item)=>item.updates);
  const committedUpdates=updateValues.length&&updateValues.every((value)=>typeof value==="number")?updateValues.reduce((sum,value)=>sum+value,0):null;
  const groupPanel=panel("Groups",groupRows.length?dataTable("group-rows",groupRows,[
    {key:"task_id",label:"Task ID",render:(row)=>label(row.task_id)},
    {key:"source_class",label:"Cohort",render:(row)=>badge(groupEvidenceView(row).sourceClass)},
    {key:"phase",label:"Phase",render:(row)=>groupEvidenceView(row).phase},
    {key:"group_id",label:"Group",render:(row)=>label(row.group_id)},
    {key:"status",label:"Group state",render:(row)=>badge(label(row.status))},
    {key:"record_status",label:"Record state",render:(row)=>badge(label(row.record_status))},
    {key:"exact",label:"Native exact",render:(row)=>groupEvidenceView(row).exact},
    {key:"rewards",label:"Native rewards",render:(row)=>groupEvidenceView(row).rewards},
    {key:"raw_group_size",label:"Raw size",numeric:true,render:(row)=>label(row.raw_group_size)},
    {key:"training_group_size",label:"Prepared rows",numeric:true,render:(row)=>groupEvidenceView(row).preparedTrainingSize},
    {key:"training_rewards",label:"Observed train rows",numeric:true,render:(row)=>groupEvidenceView(row).observedTrainingRows},
    {key:"selected_ready_cache",label:"Ready cache selected",render:(row)=>groupEvidenceView(row).selectedReadyCache},
    {key:"policy_lag",label:"Lag",render:(row)=>groupEvidenceView(row).policyLag},
    {key:"optimizer_consumed",label:"Optimizer",render:(row)=>{const item=optimizerEvidence(row);return badge(item.label,item.tone);}},
    {key:"trajectory_artifact",label:"轨迹",sortable:false,render:(row)=>row.trajectory_artifact?makeLink("打开轨迹",()=>openTrajectory(workspace.run.run_id,String(row.trajectory_artifact).split("/").pop())):NOT_RECORDED},
  ],{pageSize:14,note:"Prepared rows ≠ observed training_rewards ≠ optimizer-consumed rows; group.exact and rewards are native candidates",onRow:openGroup}):emptyState("当前 run 没有组记录","selected run 的 record.groups 中没有已记录组。"),
    {eyebrow:"OPTIMIZER GROUPS",description:"显示 canonical task_id、exact、rewards、大小与 cache 选择；optimizer_consumed=false 明确表示未消费。",flush:true});
  const fileTable=files.length?dataTable("trajectory-files",files,[
    {key:"name",label:"文件",render:(row)=>makeLink(row.name||"unknown",()=>openTrajectory(workspace.run.run_id,row.name),"text-button mono")},
    {key:"bytes",label:"大小",numeric:true,render:(row)=>fmtBytes(row.bytes)},
    {key:"mtime",label:"修改时间",render:(row)=>row.mtime?ago(row.mtime):NOT_RECORDED},
  ],{pageSize:12,note:"轨迹读取只返回摘要，不搬运 token arrays"}):emptyState("没有轨迹文件","run.artifacts.trajectories 为空。");
  return el("div",{className:"view-stack"},
    el("header",{className:"section-title"},el("div",{className:"eyebrow"},"DATA LINEAGE"),el("h2",{},"Groups & trajectories"),el("p",{},"canonical record.groups outcome、prepared rows、observed training_rewards 与 optimizer consumption 分开展示。")),
    grid([
      metric("题组记录",fmtInt(groupRows.length),"groups","所有已选 run records"),
      metric("Native exact",exactCountTotal===null?NOT_RECORDED:(fmtInt(exactCountTotal)+" / "+fmtInt(exactKnownTotal)),"known rows",exactTotals.length+" groups with group.exact evidence · 不合并 guided replay","teal"),
      metric("Optimizer 已消费",consumed,"groups","只统计 optimizer_consumed 布尔字段"),
      metric("成功更新",fmtInt(committedUpdates),"updates","所有 run records 的已知 completed update 数"),
    ],"four"),
    failuresPanel(failedRecords),
    panel("Persisted cohort coverage",cohortTable(cohortRows),{eyebrow:"SOURCE CLASS × POLICY × PHASE",description:"Prepared cardinalities are not committed rows; observed training_rewards and optimizer consumption remain separate.",flush:true}),
    groupPanel,
    grid([
      panel("轨迹文件",fileTable,{eyebrow:"RUN ARTIFACTS",description:workspace.run&&workspace.run.run_id||"selected run"}),
      panel("组奖励解释",el("div",{className:"guidance-list"},
        el("div",{},el("b",{},"Native exact"),el("p",{},"只统计 group.exact 中明确的布尔值；缺失或 null 行保持未记录。")),
        el("div",{},el("b",{},"Prepared vs observed"),el("p",{},"training_group_size / group_cardinalities 是准备规模；training_rewards 是成功更新后记录的训练向量。")),
        el("div",{},el("b",{},"Cache vs optimizer"),el("p",{},"selected_ready_cache 表示选择了 ready cache；optimizer_consumed=false 表示该组没有被 optimizer 消费。"))),{eyebrow:"COUNTING SCOPE"}),
    ],"two"),
    panel("Run record coverage",dataTable("group-records",allRecords,[
      {key:"record_id",label:"Record"},{key:"kind",label:"Kind"},{key:"status",label:"State",render:(row)=>badge(label(row.status))},
      {key:"error.type",label:"Failure",render:(row)=>row.error&&typeof row.error==="object"?label(row.error.type):row.error?"已记录":"未记录"},
      {key:"updates",label:"Updates",numeric:true},{key:"policy_version",label:"Policy",numeric:true},
      {key:"groups",label:"Groups",numeric:true,render:(row)=>safeArray(row.groups).length},
      {key:"finished_at",label:"Finished",render:(row)=>row.finished_at?fmtDate(row.finished_at):NOT_RECORDED},
    ],{pageSize:10,note:"record groups 与 files 通过 run_id 分区"}),{eyebrow:"RECORD INDEX",flush:true})
  );
}
