const $ = (id) => document.getElementById(id);
const state = { model: 'gemma3:4b', layer: 0, layers: 34, visionLayers: 27, hidden: 2560, mlp: 10240, attentionHeads: 8, kvHeads: 4, headWidth: 256, visionHidden: 1152, visionMlp: 4304, component: 'text_mlp', neuron: 0, directoryStart: 0, directorySize: 20, busy: false, image: null, history: [] };

const setText = (id, value) => { const node = $(id); if (node) node.textContent = value; };
function setStatus(label, mode = '') {
  setText('connectionText', label);
  $('connectionDot')?.classList.remove('ready', 'busy');
  if (mode) $('connectionDot')?.classList.add(mode);
}
function setFlow(label, mode = '') {
  setText('flowStatus', label);
  $('flowDot')?.classList.remove('ready', 'busy');
  if (mode) $('flowDot')?.classList.add(mode);
}
function wordsApprox(text) { return text.trim() ? text.trim().split(/\s+/u).length : 0; }

async function loadModels() {
  try {
    const r = await fetch('/api/models');
    if (!r.ok) throw new Error((await r.json()).error || `HTTP ${r.status}`);
    const data = await r.json();
    const models = data.models || [];
    const select = $('modelSelect');
    select.replaceChildren(...models.map(m => { const o = document.createElement('option'); o.value = m.name; o.textContent = m.name; return o; }));
    if (!models.length) throw new Error('Ollama has no installed models');
    state.model = models.some(m => m.name === 'gemma3:4b') ? 'gemma3:4b' : models[0].name;
    select.value = state.model;
    await loadModelInfo();
    setStatus('OLLAMA CONNECTED', 'ready');
  } catch (e) { setStatus('OLLAMA OFFLINE'); console.warn(e); }
}

async function loadModelInfo() {
  state.model = $('modelSelect').value;
  setText('topModel', state.model);
  setText('outputModel', state.model);
  try {
    const r = await fetch(`/api/model?name=${encodeURIComponent(state.model)}`);
    const info = await r.json();
    if (!r.ok) throw new Error(info.error || 'Could not inspect model');
    setText('factFamily', String(info.family || 'unknown').toUpperCase());
    setText('factParams', Number(info.parameter_count || 3880101888).toLocaleString());
    setText('factLayers', info.layers ?? 'unknown');
    setText('factQuant', info.quantization || 'unknown');
    state.layers = Number(info.layers) || 34;
    state.hidden = Number(info.embedding) || 2560;
    state.mlp = Number(info.mlp_width) || 10240;
    state.attentionHeads = Number(info.attention_heads) || 8;
    state.kvHeads = Number(info.kv_heads) || 4;
    state.headWidth = Number(info.head_width) || 256;
    state.visionLayers = Number(info.vision_layers) || 27;
    state.visionHidden = Number(info.vision_embedding) || 1152;
    state.visionMlp = Number(info.vision_mlp_width) || 4304;
    setLayer(0);
    updateNeuronSpec();
  } catch (e) { console.warn(e); }
}

function neuronSpec() {
  const specs = {
    text_mlp: {name:'Text MLP neurons', short:'MLP', prefix:'T', width:state.mlp, blocks:state.layers, kind:'neurons'},
    vision_mlp: {name:'Vision MLP neurons', short:'MLP', prefix:'V', width:state.visionMlp, blocks:state.visionLayers, kind:'neurons'},
    text_residual: {name:'Text residual channels', short:'RES', prefix:'T', width:state.hidden, blocks:state.layers, kind:'channels'},
    text_query: {name:'Text attention query channels', short:'Q', prefix:'T', width:state.attentionHeads*state.headWidth, blocks:state.layers, kind:'channels'},
    text_kv: {name:'Text attention key/value channels', short:'KV', prefix:'T', width:state.kvHeads*state.headWidth, blocks:state.layers, kind:'channels'}
  };
  return specs[state.component] || specs.text_mlp;
}
function networkSpec(view3d=false) {
  const spec=neuronSpec(),blockCount=spec.blocks,visibleBlocks=Math.min(6,blockCount);
  if(view3d){
    const textIn=state.hidden,textMlp=state.mlp,visionIn=state.visionHidden,visionMlp=state.visionMlp;
    if(state.component==='text_mlp') return {
      stages:[
        {title:'RESIDUAL IN',count:textIn,tag:'INPUT',unitShort:'RES',role:'channel'},
        {title:'PRE-FFN RMSNORM',count:textIn,tag:'PRENORM',unitShort:'PRENORM',role:'channel'},
        {title:'GATE PROJECTION',count:textMlp,tag:'GATE',unitShort:'MLP',role:'main',yOffset:-34},
        {title:'UP PROJECTION',count:textMlp,tag:'UP',unitShort:'MLP',role:'main',yOffset:34},
        {title:'TANH-GELU × UP',count:textMlp,tag:'MLP',unitShort:'MLP',role:'main'},
        {title:'DOWN PROJECTION',count:textIn,tag:'DOWN',unitShort:'DOWN',role:'channel'},
        {title:'POST-FFN RMSNORM',count:textIn,tag:'POSTNORM',unitShort:'POSTNORM',role:'channel'},
        {title:'RESIDUAL ADD',count:textIn,tag:'RES.OUT',unitShort:'RES.OUT',role:'channel'}
      ],
      edges:[[0,1,'dense'],[1,2,'dense'],[1,3,'dense'],[2,4,'paired'],[3,4,'paired'],[4,5,'dense'],[5,6,'dense'],[6,7,'paired'],[0,7,'paired']],
      details:`Text block ${state.layer+1} of ${state.layers}: ${textIn.toLocaleString()} residual channels pass through pre-FFN RMSNorm, separate ${textMlp.toLocaleString()}-wide gate/up projections, GELU-tanh(gate) × up, down projection, and post-FFN RMSNorm before the residual add. Norm edges show sampled channel dependencies; RMSNorm uses a shared reduction and per-channel scale, not a dense learned matrix. The attention sublayer is outside this view. Each stage draws 16 samples; labels give full widths.`,
      dimension:`TEXT BLOCK ${state.layer+1}/${state.layers} · ${textIn.toLocaleString()} → ${textMlp.toLocaleString()} GATED UNITS → ${textIn.toLocaleString()}`
    };
    if(state.component==='vision_mlp') return {
      stages:[
        {title:'VISION RESIDUAL IN',count:visionIn,tag:'INPUT',unitShort:'RES',role:'channel'},
        {title:'PRE-MLP LAYER NORM',count:visionIn,tag:'PRENORM',unitShort:'PRENORM',role:'channel'},
        {title:'FC1 + GELU',count:visionMlp,tag:'FC1',unitShort:'MLP',role:'main'},
        {title:'FC2 OUTPUT',count:visionIn,tag:'OUTPUT',unitShort:'RES',role:'channel'},
        {title:'RESIDUAL ADD',count:visionIn,tag:'RES.OUT',unitShort:'RES.OUT',role:'channel'}
      ],edges:[[0,1,'dense'],[1,2,'dense'],[2,3,'dense'],[3,4,'paired'],[0,4,'paired']],
      details:`Vision block ${state.layer+1} of ${state.visionLayers}: ${visionIn.toLocaleString()} channels pass through pre-MLP layer norm, FC1 + GELU with ${visionMlp.toLocaleString()} units, FC2 back to ${visionIn.toLocaleString()} channels, and the residual add. The attention sublayer is not shown. Each stage displays 16 sampled indices; widths are the configured full counts.`,
      dimension:`VISION BLOCK ${state.layer+1}/${state.visionLayers} · ${visionIn.toLocaleString()} → ${visionMlp.toLocaleString()} → ${visionIn.toLocaleString()}`
    };
    if(state.component==='text_query') return {
      stages:[
        {title:'RESIDUAL INPUT',count:textIn,tag:'INPUT',unitShort:'RES',role:'channel'},
        {title:'PRE-ATTN RMSNORM',count:textIn,tag:'PRENORM',unitShort:'PRENORM',role:'channel'},
        {title:`QUERY · ${state.attentionHeads} × ${state.headWidth}`,count:state.attentionHeads*state.headWidth,tag:'QUERY',unitShort:'Q',role:'main'}
      ],edges:[[0,1,'dense'],[1,2,'dense']],
      details:`Text block ${state.layer+1} of ${state.layers}: the ${textIn.toLocaleString()}-channel residual stream passes through pre-attention RMSNorm, then Q projects to ${state.attentionHeads} heads × ${state.headWidth} channels. K/V, attention, and output projection are outside this query-projection view. Norm links show sampled channel dependencies, not a dense learned matrix.`,
      dimension:`TEXT BLOCK ${state.layer+1}/${state.layers} · ${textIn.toLocaleString()} → ${state.attentionHeads} × ${state.headWidth} QUERY`
    };
    if(state.component==='text_kv'){
      const kvWidth=state.kvHeads*state.headWidth;
      return {
        stages:[
          {title:'RESIDUAL INPUT',count:textIn,tag:'INPUT',unitShort:'RES',role:'channel'},
          {title:'PRE-ATTN RMSNORM',count:textIn,tag:'PRENORM',unitShort:'PRENORM',role:'channel'},
          {title:`KEY · ${state.kvHeads} × ${state.headWidth}`,count:kvWidth,tag:'KEY',unitShort:'K',idShort:'KV',role:'main',yOffset:-30},
          {title:`VALUE · ${state.kvHeads} × ${state.headWidth}`,count:kvWidth,tag:'VALUE',unitShort:'V',idShort:'KV',role:'main',yOffset:30,selectionOffset:kvWidth}
        ],edges:[[0,1,'dense'],[1,2,'dense'],[1,3,'dense']],
        details:`Text block ${state.layer+1} of ${state.layers}: the ${textIn.toLocaleString()}-channel residual stream passes through pre-attention RMSNorm, then separate key and value projections to ${state.kvHeads} heads × ${state.headWidth} channels each. Attention and output projection are outside this view. Norm links show sampled channel dependencies, not a dense learned matrix.`,
        dimension:`TEXT BLOCK ${state.layer+1}/${state.layers} · ${textIn.toLocaleString()} → ${state.kvHeads} × ${state.headWidth} KEY + VALUE`
      };
    }
    return {
      stages:[
        {title:'RESIDUAL INPUT',count:textIn,tag:'INPUT',unitShort:'RES',role:'channel'},
        {title:'IDENTITY SKIP',count:textIn,tag:'OUTPUT',unitShort:'RES',role:'main'}
      ],edges:[[0,1,'paired']],
      details:`Text block ${state.layer+1} of ${state.layers}: ${textIn.toLocaleString()} residual channels with one-to-one identity skip links.`,
      dimension:`TEXT BLOCK ${state.layer+1}/${state.layers} · ${textIn.toLocaleString()} RESIDUAL CHANNELS`
    };
  }
  const firstBlock=Math.max(0,Math.min(blockCount-visibleBlocks,state.layer-Math.floor(visibleBlocks/2)));
  const blocks=Array.from({length:visibleBlocks},(_,i)=>firstBlock+i);
  const stages=[
    {title:'INPUT',count:spec.width,tag:'INPUT',role:'main',layerIndex:0},
    ...blocks.map(index=>({title:`BLOCK ${String(index+1).padStart(2,'0')}`,count:spec.width,tag:'NEURONS',role:'main',layerIndex:index})),
    {title:'OUTPUT',count:spec.width,tag:'OUTPUT',role:'main',layerIndex:blockCount-1}
  ];
  return {
    stages,
    edges:stages.slice(1).map((_,i)=>[i,i+1,'dense']),
    blocks:blockCount,visibleBlocks,firstBlock,
    dimension:`${visibleBlocks} OF ${blockCount} BLOCKS SHOWN · ${spec.width.toLocaleString()} ${spec.name.toUpperCase()} PER BLOCK`,
    details:`Layer-stack schematic showing ${visibleBlocks} representative blocks. Each column draws 16 sampled ${spec.name.toLowerCase()} and fully connects them to the next column. The links illustrate a network flow; they are not Gemma's literal inter-block weight matrices.`
  };
}
function neuronId(index=state.neuron) {
  const spec=neuronSpec();
  return `${spec.prefix}${String(state.layer+1).padStart(2,'0')}.${spec.short}.${String(index+1).padStart(5,'0')}`;
}
function renderNeuronDirectory() {
  const spec=neuronSpec(), total=spec.width;
  state.directoryStart=Math.max(0,Math.min(Math.floor((total-1)/state.directorySize)*state.directorySize,state.directoryStart));
  const end=Math.min(total,state.directoryStart+state.directorySize), rows=[];
  for(let i=state.directoryStart;i<end;i++) rows.push(`<button type="button" role="listitem" class="neuron-id-button${i===state.neuron?' selected':''}" data-index="${i}">${neuronId(i)}</button>`);
  $('neuronDirectory').innerHTML=rows.join('');
  setText('directoryRange',`${(state.directoryStart+1).toLocaleString()}–${end.toLocaleString()} / ${total.toLocaleString()}`);
  setText('directoryPage',`PAGE ${Math.floor(state.directoryStart/state.directorySize)+1} / ${Math.ceil(total/state.directorySize)}`);
}
function updateNeuronSpec() {
  const spec=neuronSpec(); state.layer=Math.min(state.layer,spec.blocks-1); state.neuron=Math.min(state.neuron,spec.width-1);
  $('layerRange').max=String(spec.blocks-1); $('layerTotal').textContent=String(spec.blocks);
  $('neuronCount').textContent=spec.width.toLocaleString();
  $('neuronCountLabel').textContent=`${spec.kind} in this ${spec.prefix==='V'?'vision block':'text block'}`;
  $('neuronTotal').textContent=`${(spec.width*spec.blocks).toLocaleString()} across ${spec.prefix==='V'?'vision':'text'} stack`;
  const graph=networkSpec();
  $('architectureCount').textContent=`${graph.stages.length} LAYERS · 16 SAMPLED NEURONS / LAYER · 16×16 LINKS / ADJACENT LAYER`;
  setText('connectionDetails',graph.details);
  $('layerScaleCenter').textContent=graph.dimension.toUpperCase();
  setText('sceneBranchLabel',spec.prefix==='V'?'VISION':'TEXT');
  const textMlpTotal=state.layers*state.mlp, visionMlpTotal=state.visionLayers*state.visionMlp;
  const textWeights=3*state.hidden*state.mlp*state.layers,visionWeights=2*state.visionHidden*state.visionMlp*state.visionLayers;
  setText('modelArchitectureFooter',`TEXT MLP ${textMlpTotal.toLocaleString()} UNITS | ${textWeights.toLocaleString()} MLP WEIGHTS; VISION ${visionMlpTotal.toLocaleString()} UNITS | ${visionWeights.toLocaleString()} MLP WEIGHTS; TOTAL ${ (textMlpTotal+visionMlpTotal).toLocaleString()} UNITS | ${(textWeights+visionWeights).toLocaleString()} MLP WEIGHTS`);
  $('tooltipDescription').textContent=spec.name;
  renderNeuronDirectory(); setLayer(state.layer); scene?.refresh();
}
function setNeuron(index) {
  const spec=neuronSpec(); state.neuron=Math.max(0,Math.min(spec.width-1,Number(index)||0));
  setText('tooltipLayer',neuronId()); $('neuronSearch').value=neuronId(); scene?.refresh();
  const pageStart=Math.floor(state.neuron/state.directorySize)*state.directorySize;
  if(pageStart!==state.directoryStart){state.directoryStart=pageStart;renderNeuronDirectory();}
  else document.querySelectorAll('.neuron-id-button').forEach(button=>button.classList.toggle('selected',Number(button.dataset.index)===state.neuron));
}
function setLayer(n) {
  const spec=neuronSpec(); state.layer=Math.max(0,Math.min(spec.blocks-1,Number(n)||0));
  $('layerRange').value=String(state.layer);
  setText('selectedLayerText',`BLOCK ${String(state.layer+1).padStart(2,'0')}`);
  setText('tooltipLayer',neuronId()); $('neuronSearch').value=neuronId(); renderNeuronDirectory(); scene?.refresh();
}

function markTrace(index, phase, detail) {
  const steps = document.querySelectorAll('.trace-step');
  steps.forEach((step, i) => step.classList.toggle('active', phase === 'active' && i === index));
  if (phase === 'complete') steps[index]?.classList.add('complete');
  if (detail) setText(['tracePrompt', 'traceLayer', 'traceOutput'][index], detail);
}
function resetTrace() {
  document.querySelectorAll('.trace-step').forEach(s => s.classList.remove('active', 'complete'));
  setText('tracePrompt', 'Waiting for input'); setText('traceLayer', `${state.layers} text blocks · structural counts`); setText('traceOutput', 'Waiting for generation');
}

function renderScene() {
  const host=$('scene'), canvas=document.createElement('canvas');
  canvas.setAttribute('aria-label','Animated layer and neuron network visualization.');
  host.appendChild(canvas);
  const ctx=canvas.getContext('2d');
  let zoom=1,panX=0,panY=0,dragging=false,lastX=0,lastY=0,layout=null,hoveredNode=null,raf=0,view3d=false,cameraYaw=-.34,cameraPitch=.18,lastInfoKey='';
  function draw() {
    const dpr=Math.min(window.devicePixelRatio||1,2),w=host.clientWidth,h=host.clientHeight;
    if(!w||!h)return;
    if(canvas.width!==Math.round(w*dpr)||canvas.height!==Math.round(h*dpr)){canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);}
    ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);
    const graph=networkSpec(view3d),spec=neuronSpec(),graphW=Math.min(900,w*.84),graphH=Math.min(400,h*.82),margin=60;
    const infoKey=`${view3d}:${state.component}:${state.layer}:${graph.details}`;
    if(infoKey!==lastInfoKey){
      lastInfoKey=infoKey;
      setText('connectionDetails',graph.details);
      $('layerScaleCenter').textContent=graph.dimension.toUpperCase();
      $('architectureCount').textContent=view3d?`${graph.stages.length} INTERNAL STAGES · 16 SAMPLED UNITS / STAGE`:`${graph.stages.length} LAYERS · 16 SAMPLED NEURONS / LAYER · 16×16 LINKS / ADJACENT LAYER`;
    }
    const visible=16,step=(graphW-margin*2)/Math.max(1,graph.stages.length-1),rowGap=Math.min(18,graphH/(visible+1));
    const rowGap3d=Math.min(64,graphH/5),depthGap=Math.min(24,graphW/32);
    const now=performance.now(),elapsed=state.busy?now-(state.runStarted||now):now;
    const activeStage=state.busy?Math.floor(elapsed/900)%graph.stages.length:-1;
    const nodesByStage=[],nodes=[];
    graph.stages.forEach((stage,si)=>{
      let indices;
      if(stage.kvWidth){indices=[0,1,2,3,stage.kvWidth,stage.kvWidth+1,stage.kvWidth+2,stage.kvWidth+3];}
      else if(stage.role==='main'&&stage.count>visible){const selectedIndex=state.neuron-(stage.selectionOffset||0);const start=Math.max(0,Math.min(stage.count-visible,selectedIndex-Math.floor(visible/2)));indices=Array.from({length:visible},(_,i)=>start+i);}
      else indices=Array.from({length:Math.min(visible,stage.count)},(_,i)=>i);
      const x=-graphW/2+margin+step*si,stageNodes=[];
      indices.forEach((index,j)=>{
        const offset=(j-(indices.length-1)/2)*rowGap*.72;
        const y=view3d?(1.5-Math.floor(j/4))*rowGap3d+(stage.yOffset||0):(stage.tag==='GATE'||stage.tag==='MLP'?-graphH*.2+offset:stage.tag==='UP'?graphH*.2+offset:(j-(indices.length-1)/2)*rowGap);
        const z=view3d?(j%4-1.5)*depthGap:0;
        let tag=stage.tag,unit=index;
        if(stage.kvWidth){tag=index<stage.kvWidth?'K':'V';unit=index%stage.kvWidth;}
        const layerIndex=stage.layerIndex??state.layer;
        const idShort=stage.idShort||stage.unitShort||spec.short;
        const idIndex=unit+(stage.selectionOffset||0);
        const id=`${spec.prefix}${String(layerIndex+1).padStart(2,'0')}.${idShort}.${String(idIndex+1).padStart(5,'0')}`;
        const nodeIndex=stage.kvWidth?index%stage.kvWidth:index;
        const node={x,y,z,index:nodeIndex,selectionIndex:nodeIndex+(stage.selectionOffset||0),layerIndex,slot:j,id,shortLabel:stage.kvWidth?`${tag}.${String(unit+1).padStart(5,'0')}`:String(unit+1).padStart(5,'0'),tag,title:stage.title,role:stage.role,stage:si};stageNodes.push(node);nodes.push(node);
      });
      nodesByStage.push(stageNodes);
    });
    const cosYaw=Math.cos(cameraYaw),sinYaw=Math.sin(cameraYaw),cosPitch=Math.cos(cameraPitch),sinPitch=Math.sin(cameraPitch);
    function project(node){
      if(!view3d)return {x:node.x,y:node.y,scale:1,depth:0};
      const x=node.x*cosYaw+node.z*sinYaw,z=-node.x*sinYaw+node.z*cosYaw;
      const y=node.y*cosPitch-z*sinPitch,depth=node.y*sinPitch+z*cosPitch;
      const scale=1400/(1400-depth);
      return {x:x*scale,y:y*scale,scale,depth};
    }
    const drawNodesByStage=nodesByStage.map(stage=>stage.map(node=>({...node,...project(node)})));
    const drawNodes=drawNodesByStage.flat();
    layout={w,h,graphW,graphH,nodes:drawNodes};
    ctx.save();ctx.translate(w/2+panX,h/2+panY);ctx.scale(zoom,zoom);
    // Animate each connection itself as a continuous wave, keeping its ends
    // attached to the neuron circles. Neurons stay still; no particles travel.
    for(const [from,to,type] of graph.edges){
      const left=drawNodesByStage[from],right=drawNodesByStage[to];
      const activeEdge=state.busy&&(to===activeStage||from===activeStage);
      const pairs=[];
      if(type==='paired') left.forEach((node,i)=>{if(right[i])pairs.push([node,right[i]]);});
      else left.forEach(a=>right.forEach(b=>pairs.push([a,b])));
      pairs.forEach(([a,b])=>{
        const dx=b.x-a.x,dy=b.y-a.y,length=Math.hypot(dx,dy)||1;
        const phase=now*.003+a.slot*.31+b.slot*.17+from*.8;
        ctx.beginPath();ctx.moveTo(a.x,a.y);
        for(let segment=1;segment<=12;segment++){
          const t=segment/12,envelope=Math.sin(Math.PI*t);
          const offset=Math.sin(t*Math.PI*4+phase)*4.2*envelope;
          ctx.lineTo(a.x+dx*t-(dy/length)*offset,a.y+dy*t+(dx/length)*offset);
        }
        ctx.strokeStyle=activeEdge?'#d5edff55':type==='paired'?'#94f3cb32':'#99b9ce2c';
        ctx.lineWidth=activeEdge?1.05:.75;ctx.stroke();
      });
    }
    graph.stages.forEach((stage,si)=>{
      const label=project({x:-graphW/2+margin+step*si,y:-graphH/2+8,z:0});
      ctx.textAlign='center';ctx.fillStyle='#9aabb8';ctx.font='8px "DM Mono", monospace';ctx.fillText(stage.title,label.x,label.y);
      ctx.fillStyle='#64798a';ctx.font='8px "DM Mono", monospace';ctx.fillText(`${stage.count.toLocaleString()} UNITS`,label.x,label.y+13*label.scale);
    });
    for(const node of drawNodes){
      const selected=node.id===hoveredNode?.id;
      ctx.beginPath();ctx.arc(node.x,node.y,(selected?5.8:4)*node.scale,0,Math.PI*2);ctx.fillStyle=selected?'#ffbd7b':'#a8c5dc';ctx.fill();ctx.strokeStyle=selected?'#ffe0b9':'#101720';ctx.lineWidth=1;ctx.stroke();
    }
    ctx.restore();
    if(hoveredNode){setText('tooltipLayer',hoveredNode.id);setText('tooltipDescription',hoveredNode.title);setText('streamStatus',hoveredNode.id);$('neuronSearch').value=hoveredNode.id;}
    else {setText('tooltipLayer',neuronId());setText('tooltipDescription',spec.name);setText('streamStatus',state.busy?'STAGE PROGRESS · ACTIVATIONS UNAVAILABLE':'LINE WAVE · IDLE');if(document.activeElement!==$('neuronSearch'))$('neuronSearch').value=neuronId();}
    if(!raf)raf=requestAnimationFrame(()=>{raf=0;draw();});
  }
  function nodeAt(e){
    if(!layout)return null;
    const rect=canvas.getBoundingClientRect(),x=(e.clientX-rect.left-layout.w/2-panX)/zoom,y=(e.clientY-rect.top-layout.h/2-panY)/zoom;
    let best=null,bestDistance=13;
    for(const node of layout.nodes){const d=Math.hypot(node.x-x,node.y-y);if(d<bestDistance){best=node;bestDistance=d;}}
    return best;
  }
  canvas.addEventListener('pointerdown',e=>{dragging=true;lastX=e.clientX;lastY=e.clientY;canvas.setPointerCapture(e.pointerId);});
  canvas.addEventListener('pointerup',()=>dragging=false);
  canvas.addEventListener('pointermove',e=>{
    if(dragging){
      const dx=e.clientX-lastX,dy=e.clientY-lastY;
      if(view3d){cameraYaw+=dx*.006;cameraPitch=Math.max(-.9,Math.min(.9,cameraPitch+dy*.006));}
      else {panX+=dx;panY+=dy;}
      lastX=e.clientX;lastY=e.clientY;draw();return;
    }
    const node=nodeAt(e);canvas.style.cursor=node?'pointer':'grab';
    if(node?.id!==hoveredNode?.id||node?.stage!==hoveredNode?.stage){hoveredNode=node;draw();}
  });
  canvas.addEventListener('pointerleave',()=>{if(!dragging&&hoveredNode){hoveredNode=null;draw();}});
  canvas.addEventListener('click',()=>{if(hoveredNode&&hoveredNode.role==='main'){setLayer(hoveredNode.layerIndex);setNeuron(hoveredNode.selectionIndex??hoveredNode.index);}});
  canvas.addEventListener('wheel',e=>{zoom=Math.max(.8,Math.min(2.8,zoom+(e.deltaY<0?.08:-.08)));draw();},{passive:true});
  $('viewToggle').addEventListener('click',()=>{
    view3d=!view3d;zoom=1;panX=0;panY=0;hoveredNode=null;
    $('viewToggle').textContent=view3d?'2D VIEW':'3D VIEW';
    $('viewToggle').setAttribute('aria-pressed',String(view3d));
    setText('sceneTitle',view3d?'GEMMA 3 · SELECTED BLOCK TOPOLOGY':'GEMMA 3 · MODEL LAYER STACK');
    canvas.setAttribute('aria-label',view3d?'Rotatable 3D layer and neuron network visualization.':'Animated 2D layer and neuron network visualization.');
    draw();
  });
  new ResizeObserver(draw).observe(host);draw();
  return {refresh(){hoveredNode=null;draw();}};
}
const scene = renderScene();

async function run() {
  if (state.busy) return;
  const prompt = $('promptInput').value.trim();
  if (!prompt) { $('promptInput').focus(); return; }
  state.busy = true; state.runStarted=performance.now(); scene.refresh(); $('runButton').disabled = true; setText('runButtonLabel','Generating…');
  setText('outputState','RUNNING'); $('outputState').className='output-state running'; setStatus('GENERATING','busy'); setFlow('GENERATING · ACTIVATIONS UNAVAILABLE','busy');
  setText('outputTokens','0'); setText('tokensPerSecond','— TOK/S');
  $('responseText').textContent=''; resetTrace();
  const combined = [...state.history, {role:'user', content:prompt}].map(m => `${m.role === 'user' ? 'User' : 'Assistant'}: ${m.content}`).join('\n\n') + '\n\nAssistant:';
  markTrace(0,'active',`${wordsApprox(prompt)} approximate words · exact tokenizer count unavailable`);
  const payload={model:state.model,prompt:combined,stream:true,options:{temperature:Number($('temperature').value),num_predict:Number($('maxTokens').value)}};
  if (state.image) payload.images=[state.image];
  const start=performance.now(); let answer='', stats=null;
  try {
    const response=await fetch('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    if (!response.ok) { const err=await response.json().catch(()=>({})); throw new Error(err.error||`HTTP ${response.status}`); }
    const reader=response.body.getReader(), decoder=new TextDecoder(); let buffer='';
    while (true) {
      const {value,done}=await reader.read(); if(done) break; buffer+=decoder.decode(value,{stream:true});
      const lines=buffer.split('\n'); buffer=lines.pop();
      for(const line of lines) { if(!line.trim()) continue; const item=JSON.parse(line); if(item.error) throw new Error(item.error);
        if(item.response) { answer+=item.response; $('responseText').textContent=answer; $('responseText').scrollTop=$('responseText').scrollHeight; markTrace(1,'active','Selected structural unit; live activations unavailable from Ollama'); markTrace(2,'active',`${wordsApprox(answer)} approximate words generated`);
        }
        if(item.done) stats=item;
      }
    }
    state.history.push({role:'user',content:prompt},{role:'assistant',content:answer});
    const elapsed=(performance.now()-start)/1000, count=stats?.eval_count;
    setText('outputTokens',count ?? wordsApprox(answer));
    $('outputTokens').parentElement.lastChild.textContent = count == null ? ' APPROX. WORDS' : ' TOKENS';
    setText('tokensPerSecond',stats?.eval_count && stats?.eval_duration ? `${(stats.eval_count/(stats.eval_duration/1e9)).toFixed(1)} TOK/S` : `${(wordsApprox(answer)/Math.max(elapsed,.1)).toFixed(1)} WORD/S*`);
    setText('outputTiming',stats?.total_duration?`${(stats.total_duration/1e9).toFixed(2)} SEC`:'LOCAL INFERENCE');
    markTrace(0,'complete',`${stats?.prompt_eval_count ?? wordsApprox(combined)} ${stats?.prompt_eval_count == null?'approx. words':'Ollama tokens'} processed`);
    markTrace(1,'complete',`${state.layers} text blocks; ${state.mlp.toLocaleString()} MLP units per block. Activation values unavailable.`);
    markTrace(2,'complete',`${count ?? wordsApprox(answer)} ${count == null?'approx. words':'Ollama tokens'} generated`);
    $('outputState').textContent='COMPLETE'; $('outputState').className='output-state done'; setFlow('TRACE COMPLETE','ready'); setStatus('OLLAMA CONNECTED','ready');
  } catch(e) {
    $('responseText').textContent=`Request failed: ${e.message}`; $('outputState').textContent='ERROR'; $('outputState').className='output-state'; setFlow('REQUEST FAILED'); setStatus('OLLAMA ERROR');
  } finally { state.busy=false; scene.refresh(); $('runButton').disabled=false; setText('runButtonLabel','Run inference'); }
}

$('runButton').addEventListener('click',run);
$('promptInput').addEventListener('keydown',e=>{if((e.ctrlKey||e.metaKey)&&e.key==='Enter') run();});
$('modelSelect').addEventListener('change',loadModelInfo);
$('temperature').addEventListener('input',e=>setText('tempValue',Number(e.target.value).toFixed(2)));
$('maxTokens').addEventListener('input',e=>setText('maxTokensValue',`${e.target.value} TOKENS`));
$('layerRange').addEventListener('input',e=>setLayer(e.target.value));
$('neuronComponent').addEventListener('change',e=>{state.component=e.target.value;state.layer=0;state.neuron=0;state.directoryStart=0;updateNeuronSpec();});
function findNeuron(){
  const raw=$('neuronSearch').value.trim();
  const full=raw.match(/^([TV])(\d+)\.(MLP|GATE|UP|FC1|FC2|RES(?:\.IN|\.OUT)?|PRENORM|POSTNORM|DOWN|Q|KV|K|V)\.(\d+)$/i);
  if(full){
    const [,branch,block,kindRaw,index]=full,kind=kindRaw.toUpperCase();
    let key='text_mlp';
    if(branch.toUpperCase()==='V') key='vision_mlp';
    else if(kind==='Q') key='text_query';
    else if(['KV','K','V'].includes(kind)) key='text_kv';
    else if(['RES','RES.IN','RES.OUT','PRENORM','POSTNORM','DOWN'].includes(kind)) key='text_residual';
    state.component=key; $('neuronComponent').value=key; state.directoryStart=0; updateNeuronSpec();
    setLayer(Number(block)-1); setNeuron(Number(index)-1); return;
  }
  const number=Number(raw.match(/\d+$/)?.[0]);
  if(Number.isFinite(number)&&number>0)setNeuron(number-1);
}
$('findNeuron').addEventListener('click',findNeuron);$('neuronSearch').addEventListener('keydown',e=>{if(e.key==='Enter')findNeuron();});
$('directoryPrev').addEventListener('click',()=>{state.directoryStart=Math.max(0,state.directoryStart-state.directorySize);renderNeuronDirectory();});
$('directoryNext').addEventListener('click',()=>{state.directoryStart+=state.directorySize;renderNeuronDirectory();});
$('neuronDirectory').addEventListener('click',e=>{const button=e.target.closest('.neuron-id-button');if(button)setNeuron(button.dataset.index);});
$('prevLayer').addEventListener('click',()=>setLayer(state.layer-1)); $('nextLayer').addEventListener('click',()=>setLayer(state.layer+1));
$('imageInput').addEventListener('change',e=>{ const file=e.target.files[0]; if(!file)return; const reader=new FileReader(); reader.onload=()=>{state.image=String(reader.result).split(',')[1]; $('imagePreview').src=reader.result; $('imagePreviewWrap').hidden=false; setText('imageLabel',file.name);}; reader.readAsDataURL(file); });
$('removeImage').addEventListener('click',()=>{state.image=null;$('imageInput').value='';$('imagePreviewWrap').hidden=true;setText('imageLabel','Attach image');});
$('clearButton').addEventListener('click',()=>{state.history=[];state.image=null;$('responseText').innerHTML='<span class="response-placeholder">Your response will appear here as the model generates it<span class="cursor">▍</span></span>'; $('imagePreviewWrap').hidden=true; setText('outputTokens','0');$('outputTokens').parentElement.lastChild.textContent=' TOKENS';setText('tokensPerSecond','— TOK/S');setText('outputState','IDLE');$('outputState').className='output-state';resetTrace();setFlow('READY TO TRACE','ready');});
setInterval(()=>{setText('clock',new Date().toLocaleTimeString());},1000);
loadModels();
