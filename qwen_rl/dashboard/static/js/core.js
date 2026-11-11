const STORE = "qwen-rl-dashboard-v2";
export const DASH = "—";
export function el(tag, attrs = {}, ...children) {
  const node = (tag === "svg" || tag === "path")
    ? document.createElementNS("http://www.w3.org/2000/svg", tag)
    : document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "className") node.setAttribute("class", value);
    else if (key === "text") node.textContent = String(value);
    else if (key === "dataset") Object.assign(node.dataset, value);
    else if (key === "style" && typeof value === "object") Object.assign(node.style, value);
    else if (/^on[A-Z]/.test(key)) node.addEventListener(key.slice(2).toLowerCase(), value);
    else if (key === "value" && "value" in node) node.value = value;
    else if (key === "checked" && "checked" in node) node.checked = value;
    else if (key === "disabled" && "disabled" in node) node.disabled = value;
    else node.setAttribute(key, String(value));
  }
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}
const PATHS = {
  overview:"M3 3h8v8H3z M13 3h8v5h-8z M13 10h8v11h-8z M3 13h8v8H3z",
  learner:"M4 19V5m0 14h16M7 15l4-4 3 2 5-6",
  workers:"M8 7a3 3 0 1 0 0-6 3 3 0 0 0 0 6z M16 7a3 3 0 1 0 0-6 3 3 0 0 0 0 6z M2 20v-2a5 5 0 0 1 10 0v2z M12 18a5 5 0 0 1 10 0v2h-8",
  groups:"M4 5h16v14H4z M8 9h8 M8 13h5 M8 16h3", bepa:"M12 3v18 M5 8h14 M5 16h14 M7 8a2 2 0 1 0 0 .01 M17 16a2 2 0 1 0 0 .01",
  evaluation:"M4 4h16v16H4z M8 9l2 2 5-5 M8 16h8", recovery:"M4 7V3m0 4h4 M4 7a8 8 0 1 1-1 8 M8 13l3 3 6-7",
  events:"M5 4h14v16H5z M8 8h8 M8 12h8 M8 16h5", refresh:"M20 7v5h-5 M4 17v-5h5 M5.5 9a7 7 0 0 1 11.8-2L20 12 M4 12l2.7 5a7 7 0 0 0 11.8-2",
  sun:"M12 3v2 M12 19v2 M3 12h2 M19 12h2 M5.6 5.6 7 7 M17 17l1.4 1.4 M18.4 5.6 17 7 M7 17l-1.4 1.4 M16 12a4 4 0 1 1-8 0 4 4 0 0 1 8 0z",
  moon:"M20 14.2A8 8 0 0 1 9.8 4 8 8 0 1 0 20 14.2z", search:"M20 20l-4.4-4.4 M17 10a7 7 0 1 1-14 0 7 7 0 0 1 14 0z", menu:"M4 6h16 M4 12h16 M4 18h16",
  close:"M6 6l12 12 M18 6 6 18", check:"m5 12 4 4L19 6", alert:"M12 3 2 21h20L12 3z M12 9v5 M12 18h.01"
};
export function icon(name, size = 18) {
  return el("svg",{viewBox:"0 0 24 24",width:size,height:size,fill:"none",stroke:"currentColor","stroke-width":1.7,"stroke-linecap":"round","stroke-linejoin":"round","aria-hidden":"true"},el("path",{d:PATHS[name]||PATHS.overview}));
}
export function prefs() { try { return JSON.parse(localStorage.getItem(STORE)||"{}"); } catch (_) { return {}; } }
export function savePref(key,value) { const state=prefs(); state[key]=value; try { localStorage.setItem(STORE,JSON.stringify(state)); } catch (_) {} }
export function tablePref(key) { return prefs()["table:"+key]||{}; }
export function saveTablePref(key,value) { savePref("table:"+key,value); }
export function navigationParams() {
  return new URLSearchParams(location.search);
}
export async function api(path,params={}) {
  const url=new URL(path,location.origin);
  const activeSource=navigationParams().get("source");
  if(typeof activeSource==="string"&&activeSource)url.searchParams.set("source",activeSource);
  for(const [key,value] of Object.entries(params)) if(value!==null&&value!==undefined&&value!=="") url.searchParams.set(key,String(value));
  const response=await fetch(url,{headers:{Accept:"application/json"}});
  let body; try { body=await response.json(); } catch (_) { body={error:"响应内容无法读取"}; }
  if(!response.ok) throw new Error(body.error||("HTTP "+response.status));
  return body;
}
export const n=(value)=>value!==null&&value!==undefined&&value!==""&&Number.isFinite(Number(value))?Number(value):null;
export const known=(value)=>n(value)!==null;
export function firstNumber(...values) { for(const value of values){const valueNum=n(value);if(valueNum!==null)return valueNum;}return null; }
export function path(object,key,fallback=null) { const value=String(key||"").split(".").reduce((item,part)=>item&&item[part]!==undefined?item[part]:undefined,object);return value===undefined?fallback:value; }
export function pick(object,keys,fallback=null) { for(const key of keys){const value=path(object,key);if(value!==null&&value!==undefined&&value!=="")return value;}return fallback; }
export function fmt(value,digits=1) { const x=n(value);return x===null?DASH:(Math.abs(x)>=1000?x.toLocaleString("en-US",{maximumFractionDigits:0}):x.toLocaleString("en-US",{maximumFractionDigits:digits})); }
export function fmtInt(value) { const x=n(value);return x===null?DASH:Math.round(x).toLocaleString("en-US"); }
export function fmtPct(value,digits=0) { const x=n(value);return x===null?DASH:x.toFixed(digits)+"%"; }
export function fmtBytes(value) { let x=n(value);if(x===null)return DASH;const units=["B","KB","MB","GB","TB"];let i=0;while(x>=1024&&i<4){x/=1024;i++;}return x.toFixed(i&&x<10?1:0)+" "+units[i]; }
export function fmtDuration(seconds) { const x=n(seconds);if(x===null)return DASH;if(x<1)return Math.round(x*1000)+" ms";if(x<60)return x.toFixed(1)+" s";const m=Math.floor(x/60);return m<60?m+" 分 "+Math.round(x%60)+" 秒":Math.floor(m/60)+" 小时 "+(m%60)+" 分"; }
export function fmtDate(value) { if(value===null||value===undefined)return DASH;const ms=typeof value==="number"?(value>1e12?value:value*1000):Date.parse(value);return Number.isFinite(ms)?new Date(ms).toLocaleString("zh-CN",{month:"2-digit",day:"2-digit",hour:"2-digit",minute:"2-digit",second:"2-digit",hour12:false}):String(value); }
export function ago(value) { const ms=typeof value==="number"?(value<1e12?value*1000:value):Date.parse(value);if(!Number.isFinite(ms))return DASH;const s=Math.max(0,Date.now()-ms)/1000;return s<60?Math.round(s)+" 秒前":s<3600?Math.round(s/60)+" 分钟前":s<86400?(s/3600).toFixed(1)+" 小时前":(s/86400).toFixed(1)+" 天前"; }
export function compact(value,length=20) { const text=value===null||value===undefined?"":String(value);return text.length>length?text.slice(0,length-1)+"…":text; }
export function asText(value,fallback=DASH) { return value===null||value===undefined||value===""?fallback:String(value); }
export function safeArray(value) { return Array.isArray(value)?value:[]; }
export function statusTone(value) { const x=String(value||"").toLowerCase();if(["running","ready","succeeded","complete","completed","passed","active","ok","true"].includes(x))return "good";if(["failed","error","aborted","dead","unhealthy","false"].includes(x))return "bad";if(["warning","unknown","stale","blocked","pending","stopped","partial","incomplete"].some((v)=>x.includes(v)))return "warn";return "neutral"; }
export function badge(label,tone=null) { return el("span",{className:"badge badge-"+(tone||statusTone(label))},el("i",{}),asText(label,"未知")); }
export function sourceNote(text,state="neutral") { const label=typeof text==="string"?text:(text&&typeof text==="object"?(text.label||text.path||text.name||text.kind||"evidence source"):"来源未提供");return el("span",{className:"source-note"},el("i",{className:"source-dot "+state}),label); }
export function metric(label,value,unit="",meta="",tone="") { return el("article",{className:"metric-card "+(tone?"metric-"+tone:"")},el("div",{className:"metric-label"},label),el("div",{className:"metric-value"},value===null||value===undefined?DASH:String(value),unit?el("span",{className:"metric-unit"},unit):null),meta?el("div",{className:"metric-meta"},meta):null); }
export function panel(title,body,opts={}) { return el("section",{className:"panel "+(opts.className||"")},el("div",{className:"panel-heading"},el("div",{},opts.eyebrow?el("div",{className:"eyebrow"},opts.eyebrow):null,el("h2",{},title),opts.description?el("p",{className:"panel-description"},opts.description):null),opts.meta?el("div",{className:"panel-meta"},opts.meta):null),el("div",{className:"panel-body "+(opts.flush?"flush":"")},body)); }
export function grid(children,cols="two",extra="") { return el("div",{className:"grid grid-"+cols+(extra?" "+extra:"")},children); }
export function emptyState(title,detail="") { return el("div",{className:"empty-state"},el("span",{className:"empty-mark"},"—"),el("div",{className:"empty-title"},title),detail?el("p",{},detail):null); }
export function kvRows(items) { return el("dl",{className:"kv-list"},items.filter((x)=>x&&x.length>1).flatMap(([label,value])=>[el("dt",{},label),el("dd",{},value===null||value===undefined||value===""?DASH:(value instanceof Node?value:String(value)))])); }
export function jsonDetails(title,value,max=18000) { let text;try{text=JSON.stringify(value,null,2);if(text.length>max)text=text.slice(0,max)+"\n… 已截断";}catch(_){text=String(value);}const details=el("details",{className:"json-details"},el("summary",{},title),el("pre",{},text||DASH));const key="expanded:"+title;details.open=Boolean(prefs()[key]);details.addEventListener("toggle",()=>savePref(key,details.open));return details; }
export function sectionTitle(kicker,title,description="") { return el("header",{className:"section-title"},el("div",{className:"eyebrow"},kicker),el("h2",{},title),description?el("p",{},description):null); }
export function openDrawer(title,content,kicker="运行证据") { const backdrop=document.getElementById("drawer-backdrop"),drawer=document.getElementById("detail-drawer");if(!backdrop||!drawer)return;drawer.replaceChildren(el("header",{className:"drawer-header"},el("div",{},el("div",{className:"eyebrow"},kicker),el("h2",{},title)),el("button",{className:"icon-button",type:"button",title:"关闭详情","aria-label":"关闭详情",onClick:closeDrawer},icon("close"))),el("div",{className:"drawer-content"},content));backdrop.hidden=false;drawer.classList.add("open");drawer.setAttribute("aria-hidden","false");document.body.classList.add("drawer-open"); }
export function closeDrawer() { const b=document.getElementById("drawer-backdrop"),d=document.getElementById("detail-drawer");if(b)b.hidden=true;if(d){d.classList.remove("open");d.setAttribute("aria-hidden","true");}document.body.classList.remove("drawer-open"); }
function valueFor(row,col) { return col.sortValue?col.sortValue(row):(col.key?path(row,col.key):null); }
export function dataTable(key,rows,columns,opts={}) {
  const data=safeArray(rows),saved=tablePref(key),size=opts.pageSize||12;let query=saved.search||"",sort=saved.sort||"",direction=saved.direction||"desc",page=Math.max(0,Number(saved.page)||0);
  const root=el("div",{className:"table-shell"}),toolbar=el("div",{className:"table-toolbar"}),count=el("span",{className:"table-count"}),scroll=el("div",{className:"table-scroll",dataset:{scrollKey:"table-"+key}}),table=el("table",{className:"data-table"}),head=el("thead"),body=el("tbody"),paging=el("div",{className:"table-pagination"});
  const search=el("label",{className:"table-search"},icon("search",16),el("input",{type:"search",value:query,placeholder:opts.searchPlaceholder||"搜索当前列表","aria-label":"搜索当前列表"}));
  if(opts.search!==false){toolbar.append(search);search.querySelector("input").addEventListener("input",(event)=>{query=event.target.value;page=0;draw();});}
  if(opts.note)toolbar.append(el("span",{className:"table-note"},opts.note));toolbar.append(count);
  const trh=el("tr");
  for(const col of columns){const th=el("th",{className:col.numeric?"num":""});const b=el("button",{type:"button",className:"sort-button "+(col.numeric?"num":""),onClick:()=>{if(col.sortable===false)return;if(sort===col.key){direction=direction==="asc"?"desc":"asc";}else{sort=col.key||col.label;direction=col.numeric?"desc":"asc";}page=0;draw();}},col.label,sort===col.key?el("span",{className:"sort-indicator"},direction==="asc"?"↑":"↓"):null);th.append(col.sortable===false?el("span",{className:"sort-button plain"},col.label):b);trh.append(th);}
  head.append(trh);table.append(head,body);scroll.append(table);root.append(toolbar,scroll,paging);
  function draw(){
    const needle=query.trim().toLocaleLowerCase();let filtered=data.filter((row)=>!needle||columns.some((col)=>String(valueFor(row,col)??"").toLocaleLowerCase().includes(needle))||JSON.stringify(row).toLocaleLowerCase().includes(needle));
    if(opts.filter)filtered=filtered.filter(opts.filter);
    const column=columns.find((col)=>col.key===sort||col.label===sort);
    if(column)filtered=filtered.slice().sort((a,b)=>{const av=valueFor(a,column),bv=valueFor(b,column);if(av==null)return 1;if(bv==null)return -1;const d=column.numeric?Number(av)-Number(bv):String(av).localeCompare(String(bv),"zh-CN",{numeric:true,sensitivity:"base"});return direction==="asc"?d:-d;});
    const pages=Math.max(1,Math.ceil(filtered.length/size));page=Math.min(page,pages-1);const start=page*size;const visible=filtered.slice(start,start+size);count.textContent=filtered.length.toLocaleString("en-US")+" 条";body.replaceChildren();
    if(!visible.length){body.append(el("tr",{},el("td",{colspan:columns.length,className:"empty-cell"},opts.emptyTitle||"没有匹配记录")));}
    for(const row of visible){const tr=el("tr",{className:opts.onRow?"row-action":""});for(const col of columns){const value=col.render?col.render(row):valueFor(row,col);tr.append(el("td",{className:col.numeric?"num":""},value===null||value===undefined||value===""?DASH:(value instanceof Node?value:String(value))));}if(opts.onRow){tr.tabIndex=0;tr.addEventListener("click",(event)=>{if(event.target.closest("button,a,input,select,details"))return;opts.onRow(row);});tr.addEventListener("keydown",(event)=>{if(event.key==="Enter")opts.onRow(row);});}body.append(tr);}
    const prev=el("button",{type:"button",className:"page-button",disabled:page===0,onClick:()=>{page=Math.max(0,page-1);draw();}},"上一页"),next=el("button",{type:"button",className:"page-button",disabled:page>=pages-1,onClick:()=>{page=Math.min(pages-1,page+1);draw();}});
    paging.replaceChildren(el("span",{className:"pagination-label"},filtered.length?(start+1)+"–"+Math.min(start+size,filtered.length)+" / "+filtered.length:"0 / 0"),el("div",{className:"pagination-actions"},prev,el("span",{className:"page-number"},(page+1)+" / "+pages),next));
    saveTablePref(key,{search:query,sort,direction,page});
  }
  draw();return root;
}
export function makeLink(label,action,className="text-button") { return el("button",{type:"button",className,onClick:action},label); }
