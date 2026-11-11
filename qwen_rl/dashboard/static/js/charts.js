let activeCharts = [];
export function disposeCharts() {
  for (const item of activeCharts.splice(0)) {
    try { item.observer.disconnect(); } catch (_) {}
    try { item.chart.dispose(); } catch (_) {}
  }
}
const palette = (dark) => dark
  ? { text:"#d7e0e5", muted:"#86949a", grid:"rgba(173,193,199,.12)", tooltip:"#172328", accent:"#56b6a9", blue:"#78a6c8", amber:"#d9ad65", red:"#d98479" }
  : { text:"#253b42", muted:"#77888c", grid:"rgba(43,73,78,.12)", tooltip:"#ffffff", accent:"#127e73", blue:"#537f9e", amber:"#a97725", red:"#bc5d54" };
function startChart(element, dark) {
  if (!window.echarts || !element) return null;
  const chart = window.echarts.init(element, null, { renderer:"canvas", useDirtyRect:true });
  let timer = 0;
  const observer = new ResizeObserver(() => { clearTimeout(timer); timer=setTimeout(()=>chart.resize(),80); });
  observer.observe(element);
  activeCharts.push({chart,observer});
  return { chart, colors:palette(dark) };
}
export function mountHistoryChart(element, samples, dark) {
  const context = startChart(element,dark);
  if (!context) { element.textContent="图表资源不可用"; return; }
  const rows = Array.isArray(samples) ? samples : [];
  const toStamp=(value)=>{
    if(value===null||value===undefined)return null;
    const stamp=typeof value==="number"?(value<1e12?value*1000:value):Date.parse(value);
    return Number.isFinite(stamp)?stamp:null;
  };
  const timeOf=(row)=>toStamp(row.sampled_at);
  const observed=rows.filter((row)=>timeOf(row)!==null);
  const timeLabel=(value)=>new Date(value).toLocaleTimeString("zh-CN",{hour:"2-digit",minute:"2-digit",second:"2-digit",hour12:false});
  const baseFields=[
    ["native_decode_tokens_per_second","Native 解码（含在途）",context.colors.accent],
    ["output_tokens_per_second","已校验输出",context.colors.muted],
  ];
  const series=baseFields.map(([key,name,color])=>({
    name,type:"line",smooth:false,showSymbol:false,connectNulls:false,
    lineStyle:{width:2,color},itemStyle:{color},
    emphasis:{focus:"series"},data:observed.flatMap((row,index)=>{
      const value=typeof row[key]==="number"&&Number.isFinite(row[key])?row[key]:null;
      const previous=observed[index-1],stamp=timeOf(row);
      const gap=previous&&(stamp-timeOf(previous)>12000||previous.series_epoch!==row.series_epoch);
      return gap&&value!==null&&typeof previous[key]==="number"&&Number.isFinite(previous[key])
        ? [[timeOf(previous)+(stamp-timeOf(previous))/2,null],[stamp,value]] : [[stamp,value]];
    }),
  })).filter((item)=>item.data.some((value)=>value[1]!==null));
  const executorIds=[...new Set(observed.flatMap((row)=>
    row.executor_rates&&typeof row.executor_rates==="object"?Object.keys(row.executor_rates):[]))].sort();
  const rateColors=[context.colors.blue,context.colors.amber,context.colors.red,context.colors.accent,context.colors.muted];
  for(const [executorIndex,executorId] of executorIds.entries()){
    for(const phase of ["forward","backward"]){
      const points=[];let previous=null;const seen=new Set();
      for(const row of observed){
        const sample=row.executor_rates&&row.executor_rates[executorId];
        const measurement=sample&&sample.measurements&&sample.measurements[phase];
        if(!measurement||typeof measurement.tokens_per_second!=="number"||!Number.isFinite(measurement.tokens_per_second))continue;
        const stamp=toStamp(measurement.completed_at_unix_seconds);
        if(stamp===null||typeof measurement.sequence!=="number"||!Number.isInteger(measurement.sequence))continue;
        const epoch=typeof measurement.series_epoch==="string"?measurement.series_epoch:"";
        const key=epoch+":"+measurement.sequence;
        if(seen.has(key))continue;
        seen.add(key);
        if(previous&&(stamp-previous.stamp>12000||epoch!==previous.epoch))
          points.push([previous.stamp+(stamp-previous.stamp)/2,null]);
        points.push([stamp,measurement.tokens_per_second]);
        previous={stamp,epoch};
      }
      if(!points.some((point)=>point[1]!==null))continue;
      const color=rateColors[(executorIndex*2+(phase==="backward"?1:0))%rateColors.length];
      series.push({name:executorId+" "+(phase==="forward"?"FWD":"BWD"),type:"line",smooth:false,
        showSymbol:true,symbolSize:5,connectNulls:false,lineStyle:{width:2,color},itemStyle:{color},
        emphasis:{focus:"series"},data:points});
    }
  }
  const c=context.colors;
  context.chart.setOption({
    animation:false,
    color:series.map((item)=>item.lineStyle.color),
    grid:{left:50,right:20,top:34,bottom:34,containLabel:false},
    legend:{type:"scroll",top:0,left:0,itemWidth:12,itemHeight:7,textStyle:{color:c.muted,fontSize:11}},
    tooltip:{trigger:"axis",backgroundColor:c.tooltip,borderColor:c.grid,textStyle:{color:c.text,fontSize:11},formatter:(items)=>{
      if(!items.length)return "";
      const lines=items.map((item)=>item.marker+item.seriesName+": "+(item.value[1]===null||item.value[1]===undefined?"未知":Number(item.value[1]).toLocaleString("en-US",{maximumFractionDigits:1})+" tok/s"));
      return "<div>"+timeLabel(items[0].value[0])+"</div>"+lines.join("<br>");
    }},
    xAxis:{type:"time",minInterval:1000,
      min:observed.length?timeOf(observed[0])-1000:undefined,
      max:observed.length?Math.max(Date.now(),timeOf(observed[observed.length-1])):undefined,
      axisLine:{lineStyle:{color:c.grid}},axisTick:{show:false},
      axisLabel:{color:c.muted,fontSize:10,hideOverlap:true,formatter:timeLabel}},
    yAxis:{type:"value",min:0,axisLine:{show:false},axisTick:{show:false},splitLine:{lineStyle:{color:c.grid}},axisLabel:{color:c.muted,fontSize:10}},
    series,
    graphic:series.length?[]:[{type:"text",left:"center",top:"middle",style:{text:"所选时段没有已知吞吐样本",fill:c.muted,fontSize:12}}],
  },true);
}
