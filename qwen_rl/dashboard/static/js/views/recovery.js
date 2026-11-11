import {DASH,ago,badge,dataTable,el,emptyState,fmtBytes,fmtDate,fmtInt,grid,jsonDetails,kvRows,metric,openDrawer,panel,safeArray,sourceNote} from "../core.js";

function recoverySummary(latest){
  if(!latest)return emptyState("没有持久训练快照","不从存在 adapter 或模型文件推断 optimizer / scheduler / RNG 已保存。");
  const files=latest.files||{};
  const completeness=files.complete===true?"Required files and sizes checked":files.complete===false?"Incomplete state":"Unknown completeness";
  return el("div",{className:"recovery-summary"},
    el("div",{className:"recovery-state"},badge(completeness,files.complete===true?"good":files.complete===false?"warn":"neutral"),el("span",{},"Metadata / size audit; SHA disabled")),
    el("div",{className:"recovery-grid"},[
      ["Lineage",latest.lineage_id],["Snapshot ID",latest.snapshot_id],["Durable policy",latest.policy_version],["Completed updates",latest.completed_updates],["Engine epoch",latest.engine_epoch],["Created",latest.created_at?ago(latest.created_at):null],["Content checksum",latest.content_verification],["Size check",files.size_verification],["Parent lineage",latest.parent_lineage]
    ].map(([label,value])=>el("div",{className:"recovery-item"},el("span",{},label),el("b",{className:String(label).includes("SHA")?"mono":""},value===null||value===undefined?DASH:String(value))))),
    el("div",{className:"file-checks"},el("b",{},"Required files"),el("span",{},safeArray(files.present).length+" present / "+safeArray(files.required).length+" required"),
      files.missing&&files.missing.length?el("span",{className:"missing-files"},"Missing: "+files.missing.join(", ")):el("span",{className:"complete-files"},safeArray(files.size_mismatches).length?"Size mismatch: "+files.size_mismatches.join(", "):files.complete===true?"All required files and sizes present":"Size inventory unavailable")),
    jsonDetails("Update context",latest.update_context||{}));
}
function versionRows(recovery){return safeArray(recovery.checkpoints);}
function inventorySummary(inventory){
  const tiers=Object.entries(inventory||{}).filter(([,value])=>value&&typeof value==="object").map(([tier,value])=>({tier,expected_count:value.expected_count,present_complete_count:value.present_complete_count,bytes:value.bytes,missing_versions:safeArray(value.missing_versions),basis:value.basis}));
  const expected=tiers.length&&tiers.every((row)=>typeof row.expected_count==="number")?tiers.reduce((sum,row)=>sum+row.expected_count,0):null;
  const present=tiers.length&&tiers.every((row)=>typeof row.present_complete_count==="number")?tiers.reduce((sum,row)=>sum+row.present_complete_count,0):null;
  const missing=tiers.length?tiers.reduce((sum,row)=>sum+row.missing_versions.length,0):null;
  const bytes=tiers.length&&tiers.every((row)=>typeof row.bytes==="number")?tiers.reduce((sum,row)=>sum+row.bytes,0):null;
  return {tiers,expected,present,missing,bytes};
}
function lineageBlock(lineage){
  if(!lineage)return emptyState("Lineage metadata unavailable");
  return el("div",{className:"lineage-tree"},
    el("div",{className:"lineage-node current"},el("span",{},"CURRENT"),el("b",{},lineage.lineage_id||lineage.id||"lineage"),lineage.status?badge(lineage.status):null),
    safeArray(lineage.parents||lineage.ancestors).map((item)=>el("div",{className:"lineage-node parent"},el("span",{},"PARENT"),el("b",{},item.lineage_id||item.id||String(item)),item.status?badge(item.status):null)),
    jsonDetails("Lineage evidence",lineage));
}
export function renderRecovery(workspace){
  const recovery=workspace.recovery||{};
  if(!recovery.available)return el("div",{className:"view-stack"},el("header",{className:"section-title"},el("div",{className:"eyebrow"},"DURABLE TRAINING STATE"),el("h2",{},"Recovery & checkpoints"),el("p",{},"完整训练 state、optimizer cursor 与 checkpoint inventory。")),el("div",{className:"notice notice-info"},el("span",{className:"notice-mark"},"i"),el("div",{},el("b",{},"Recovery evidence unavailable"),el("p",{},recovery.reason||"当前来源未提供持久恢复状态。"))),recovery.source?jsonDetails("Recovery source",recovery.source):null);
  const latest=recovery.latest||null,inventory=recovery.inventory||{},checkpoints=safeArray(recovery.checkpoints),versions=versionRows(recovery),inventoryStats=inventorySummary(inventory);
  const stateComplete=latest&&latest.files&&latest.files.complete===true;
  const table=versions.length?dataTable("recovery-checkpoints",versions,[
    {key:"version",label:"Checkpoint step",numeric:true,render:(row)=>"v"+row.version},
    {key:"tier",label:"Retention tier",numeric:true},
    {key:"exists",label:"Directory",render:(row)=>row.exists===true?badge("present","good"):row.exists===false?badge("missing","bad"):badge("unknown")},
    {key:"complete",label:"Adapter files",render:(row)=>row.complete===true?badge("complete","good"):row.complete===false?badge("incomplete","warn"):badge("unknown")},
    {key:"bytes",label:"Size",numeric:true,render:(row)=>fmtBytes(row.bytes)},
    {key:"files",label:"Observed files",render:(row)=>Object.keys(row.files||{}).join(", ")||DASH},
  ],{pageSize:10,note:"Adapter checkpoints contain adapter files only; optimizer resume depends on the separate latest full-state snapshot.",onRow:(row)=>openCheckpoint(row)}):emptyState("No retained adapter checkpoints","Checkpoint retention inventory is unavailable for this selected state.");
  const tierTable=inventoryStats.tiers.length?dataTable("recovery-tier-inventory",inventoryStats.tiers,[
    {key:"tier",label:"Retention tier"},{key:"expected_count",label:"Expected",numeric:true},{key:"present_complete_count",label:"Complete adapters",numeric:true},{key:"missing_versions",label:"Missing versions",render:(row)=>row.missing_versions.length?row.missing_versions.map((value)=>"v"+value).join(", "):"—"},{key:"bytes",label:"Bytes",numeric:true,render:(row)=>fmtBytes(row.bytes)},
  ],{search:false,pageSize:5,note:"Each tier follows the persisted retention policy basis"}):emptyState("Tier inventory unknown","No retained-version tier counts were provided.");
  const titleStatus=stateComplete?"Required files and sizes checked":latest&&latest.files&&latest.files.complete===false?"Full state incomplete":"Completeness unknown";
  return el("div",{className:"view-stack"},
    el("header",{className:"section-title"},el("div",{className:"eyebrow"},"DURABLE TRAINING STATE"),el("h2",{},"Recovery & checkpoints"),el("p",{},"只将完整、文件齐备的 snapshot 标记为可恢复训练状态；adapter-only 不代表 optimizer resume。")),
    el("div",{className:"notice "+(stateComplete?"notice-success":"notice-warn")},el("span",{className:"notice-mark"},stateComplete?"✓":"!"),el("div",{},el("b",{},titleStatus),el("p",{},stateComplete?"训练状态快照包含所需文件，仍须遵循相容性与 run lineage 约束。":"当前 evidence 未证明完整 optimizer、scheduler、scaler、RNG 与 cursor 状态齐备。"))),
    grid([
      metric("Latest durable policy",latest&&latest.policy_version!==undefined?fmtInt(latest.policy_version):DASH,"version","snapshot metadata"),
      metric("Expected checkpoints",fmtInt(inventoryStats.expected),"adapter versions","tiered retention inventory"),
      metric("Complete adapter dirs",fmtInt(inventoryStats.present),"directories",inventoryStats.missing!==null?fmtInt(inventoryStats.missing)+" missing versions":"missing count unknown"),
      metric("Adapter bytes",fmtBytes(inventoryStats.bytes),"","stat-only inventory"),
    ],"four"),
    panel("Latest full-state snapshot",recoverySummary(latest),{eyebrow:"TRAINING RESUME",description:"保存证据与模型兼容性检查分开显示。"}),
    grid([
      panel("Adapter checkpoint inventory",el("div",{className:"drawer-stack"},table,tierTable),{eyebrow:"2 / 10 / 50 STEP RETENTION",description:"These checkpoints are adapter files. The latest full-state snapshot is audited separately above.",flush:true}),
      panel("Lineage",lineageBlock(recovery.lineage||recovery.latest&&recovery.latest.lineage),{eyebrow:"RUN CONTINUITY",description:"run lineage 与 policy version 保留原始来源。"})
    ],"two"),
    panel("Recovery source",jsonDetails("Source and inventory",recovery.source||{inventory:recovery.inventory,checkpoints:recovery.checkpoints}),{eyebrow:"SOURCE EVIDENCE"}),
    recovery.selected_run?panel("Selected run lineage",kvRows([["Run lineage",recovery.selected_run.lineage_id],["Matches latest",recovery.selected_lineage_matches_latest],["Run engine epoch",recovery.selected_run.engine_epoch],["Run policy",recovery.selected_run.policy_version],["Loaded snapshot",recovery.selected_run.loaded_snapshot]]),{eyebrow:"LINEAGE MATCH"}):null,
    latest&&latest.update_context?panel("恢复位置",jsonDetails("Update cursor",latest.update_context),{eyebrow:"TRAINING CURSOR"}):null
  );
}
function openCheckpoint(row){
  openDrawer("Checkpoint v"+row.version,el("div",{className:"drawer-stack"},kvRows([["Exists",row.exists],["Complete adapter files",row.complete],["Tier",row.tier],["Bytes",fmtBytes(row.bytes)],["Files",Object.keys(row.files||{}).join(", ")],["Source directory",row.source]]),jsonDetails("Checkpoint metadata",row)),"CHECKPOINT");
}
