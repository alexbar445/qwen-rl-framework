/** Snapshot immediately before replacing DOM; restoration must never animate. */
export function captureViewport() {
  const position={window:window.scrollY,left:window.scrollX,panes:{}};
  document.querySelectorAll('[data-scroll-key]').forEach((node)=>{
    position.panes[node.dataset.scrollKey]={top:node.scrollTop,left:node.scrollLeft};
  });
  return position;
}
export function restoreViewport(position={}) {
  const top=position.window||0,left=position.left||0;
  if(window.scrollY!==top||window.scrollX!==left)window.scrollTo({top,left,behavior:'instant'});
  document.querySelectorAll('[data-scroll-key]').forEach((node)=>{
    const saved=position.panes&&position.panes[node.dataset.scrollKey];
    if(saved)node.scrollTo({top:saved.top,left:saved.left,behavior:'instant'});
  });
}
