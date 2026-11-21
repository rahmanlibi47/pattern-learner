import { useCallback, useEffect, useMemo, useState } from 'react'

type Model = {
  id: string; status: string; parent_id?: string | null; created_at: string
  completed_at?: string | null; epoch: number; total_epochs: number
  train_loss?: number | null; validation_loss?: number | null; test_loss?: number | null
  train_accuracy?: number | null; validation_accuracy?: number | null; test_accuracy?: number | null
  parameter_count?: number | null; source_count: number; source_ids: string[]; error?: string | null
}
type Source = { id: string; title: string; url?: string; license?: string; license_url?: string; topic?: string; split: string; char_count: number; collected_at: string }
type Status = { status?: string; stage?: string; message?: string; current_model_id?: string; current_epoch?: number; total_epochs?: number; progress?: number; train_loss?: number; validation_loss?: number; train_accuracy?: number; validation_accuracy?: number; counts?: {articles:number;ready_models:number;characters:number}; last_error?: string }
type EventItem = { id: number; type: string; data: Record<string, unknown>; created_at: string }

const API = ''
const fmt = (value?: number | null, digits = 4) => value == null ? '—' : value.toFixed(digits)
const date = (value?: string | null) => value ? new Date(value).toLocaleString() : '—'

export default function App() {
  const [tab, setTab] = useState<'overview'|'models'|'sources'|'playground'>('overview')
  const [status, setStatus] = useState<Status>({})
  const [models, setModels] = useState<Model[]>([])
  const [sources, setSources] = useState<Source[]>([])
  const [sourceModelFilter, setSourceModelFilter] = useState('all')
  const [events, setEvents] = useState<EventItem[]>([])
  const [selectedModel, setSelectedModel] = useState('')
  const [prompt, setPrompt] = useState('the science')
  const [temperature, setTemperature] = useState(0.8)
  const [maxTokens, setMaxTokens] = useState(120)
  const [output, setOutput] = useState('')
  const [generating, setGenerating] = useState(false)
  const [error, setError] = useState('')
  const [apiConnected, setApiConnected] = useState(false)

  const refresh = useCallback(async () => {
    try {
      const [s, m, d] = await Promise.all([
        fetch(`${API}/api/status`).then(r => r.json()),
        fetch(`${API}/api/models`).then(r => r.json()),
        fetch(`${API}/api/sources?limit=1000`).then(r => r.json()),
      ])
      setStatus(s)
      setApiConnected(true)
      setError('')
      setModels(m.items ?? [])
      setSources(d.items ?? [])
      setSelectedModel(current => current || (m.items ?? []).find((x: Model) => x.status === 'ready')?.id || '')
    } catch { setApiConnected(false); setError('Cannot reach the API. Check that the Docker services are running.') }
  }, [])

  useEffect(() => {
    void refresh()
    const timer = window.setInterval(refresh, 5000)
    const stream = new EventSource(`${API}/api/events`)
    const onEvent = (event: MessageEvent) => {
      try {
        const item = { id: Number(event.lastEventId || Date.now()), type: event.type, data: JSON.parse(event.data), created_at: new Date().toISOString() }
        setEvents(current => [item, ...current].slice(0, 80))
        void refresh()
      } catch { /* Ignore malformed event payloads. */ }
    }
    ;['status','log','model'].forEach(name => stream.addEventListener(name, onEvent as EventListener))
    return () => { window.clearInterval(timer); stream.close() }
  }, [refresh])

  const readyModels = useMemo(() => models.filter(m => m.status === 'ready'), [models])
  const visibleSources = useMemo(() => {
    if (sourceModelFilter === 'all') return sources
    const model = models.find(item => item.id === sourceModelFilter)
    if (!model) return []
    const ids = new Set(model.source_ids || [])
    return sources.filter(source => ids.has(source.id))
  }, [models, sources, sourceModelFilter])
  const runGenerate = async () => {
    setGenerating(true); setError(''); setOutput('')
    try {
      const response = await fetch(`${API}/api/generate`, {
        method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ model_id: selectedModel, prompt, temperature, max_new_tokens: maxTokens }),
      })
      const body = await response.json()
      if (!response.ok) throw new Error(body.detail || 'Generation failed')
      setOutput(body.generated_text)
    } catch (e) { setError(e instanceof Error ? e.message : 'Generation failed') }
    finally { setGenerating(false) }
  }

  const progress = Math.max(0, Math.min(100, status.progress ?? 0))
  const statusLabel = status.status ?? 'connecting'

  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand"><div className="brand-mark">P</div><div><strong>Pattern Learner</strong><span>SELF-TRAINING LAB</span></div></div>
      <div className="nav-label">WORKSPACE</div>
      {([['overview','▦','Overview'],['models','◈','Model registry'],['sources','◎','Data sources'],['playground','⌘','Prompt playground']] as const).map(([key,icon,label]) =>
        <button key={key} className={`nav-item ${tab===key?'active':''}`} onClick={()=>setTab(key)}><span>{icon}</span>{label}</button>)}
      <div className="sidebar-bottom"><span className={`pulse ${(statusLabel==='training'||statusLabel==='discovering')?'running':''}`}></span><div><strong>{statusLabel.replace('_',' ')}</strong><small>Worker status</small></div></div>
    </aside>

    <main className="main">
      <header className="topbar"><div><span className="eyebrow">LOCAL INSTANCE</span><h1>{tab==='overview'?'Training overview':tab==='models'?'Model registry':tab==='sources'?'Training sources':'Prompt playground'}</h1></div><div className="top-actions"><span className={`connection ${apiConnected?'connected':'disconnected'}`}><i></i> {apiConnected?'API connected':'API offline'}</span><button className="icon-button" onClick={()=>void refresh()} title="Refresh">↻</button></div></header>
      {error && <div className="alert">{error}<button onClick={()=>setError('')}>×</button></div>}

      {tab==='overview' && <>
        <section className="hero-card"><div><div className="hero-kicker"><span className="live-dot"></span> AUTONOMOUS TRAINING WORKER</div><h2>{status.message || 'Waiting for the first training cycle'}</h2><p>Data discovery, incremental training and model publishing run independently from generation.</p></div><div className="hero-status"><span className={`status-pill ${statusLabel}`}>{statusLabel.replace('_',' ')}</span><strong>{status.current_model_id || '—'}</strong><small>{status.stage || 'initializing'}</small></div></section>
        <section className="stats-grid">
          <Stat label="READY MODELS" value={status.counts?.ready_models ?? readyModels.length} sub="Available for generation" />
          <Stat label="ARTICLES COLLECTED" value={status.counts?.articles ?? sources.length} sub={`${(status.counts?.characters ?? 0).toLocaleString()} characters stored`} />
          <Stat label="CURRENT EPOCH" value={status.total_epochs ? `${status.current_epoch ?? 0}/${status.total_epochs}` : '—'} sub={status.current_model_id || 'No active candidate'} />
          <Stat label="PARAMETER CEILING" value="1,000" sub="Default model uses 521 parameters" />
        </section>
        <section className="panel progress-panel"><div className="panel-heading"><div><span className="eyebrow">LIVE TRAINING</span><h3>{status.current_model_id || 'No candidate training'}</h3></div><strong className="progress-number">{progress.toFixed(1)}%</strong></div><div className="progress-track"><div style={{width:`${progress}%`}} /></div><div className="metric-row"><Metric label="Training loss" value={fmt(status.train_loss)} /><Metric label="Validation loss" value={fmt(status.validation_loss)} /><Metric label="Training accuracy" value={status.train_accuracy==null?'—':`${fmt(status.train_accuracy,2)}%`} /><Metric label="Validation accuracy" value={status.validation_accuracy==null?'—':`${fmt(status.validation_accuracy,2)}%`} /></div></section>
        <section className="two-columns"><div className="panel"><div className="panel-heading"><div><span className="eyebrow">LATEST VERSIONS</span><h3>Model registry</h3></div><button className="text-button" onClick={()=>setTab('models')}>View all →</button></div><ModelTable models={models.slice(0,5)} /></div><div className="panel"><div className="panel-heading"><div><span className="eyebrow">RECENT ACTIVITY</span><h3>Worker events</h3></div></div><EventList events={events.slice(0,8)} /></div></section>
      </>}

      {tab==='models' && <section className="panel page-panel"><div className="panel-heading"><div><span className="eyebrow">VERSION HISTORY</span><h3>Published and candidate models</h3></div><span className="subtle">{readyModels.length} ready / {models.length} total</span></div><ModelTable models={models} detailed /></section>}

      {tab==='sources' && <section className="panel page-panel"><div className="panel-heading"><div><span className="eyebrow">CORPUS INVENTORY</span><h3>What the learner has trained from</h3></div><select className="source-filter" value={sourceModelFilter} onChange={e=>setSourceModelFilter(e.target.value)}><option value="all">All collected sources</option>{models.map(model=><option key={model.id} value={model.id}>{model.id} · {model.status}</option>)}</select></div><div className="table-wrap"><table><thead><tr><th>ARTICLE</th><th>TOPIC</th><th>SPLIT</th><th>CHARACTERS</th><th>LICENSE</th><th>COLLECTED</th></tr></thead><tbody>{visibleSources.map(source=><tr key={source.id}><td><a href={source.url} target="_blank" rel="noreferrer">{source.title} ↗</a></td><td>{source.topic || '—'}</td><td><span className={`split-pill ${source.split}`}>{source.split}</span></td><td>{source.char_count.toLocaleString()}</td><td>{source.license_url?<a href={source.license_url} target="_blank" rel="noreferrer">{source.license || 'License'} ↗</a>:source.license||'—'}</td><td>{date(source.collected_at)}</td></tr>)}</tbody></table>{visibleSources.length===0&&<Empty text={sources.length===0?"No sources collected yet. The worker will discover licensed articles after startup.":"No sources are associated with this model in the currently loaded source list."}/>}</div><p className="footnote">Article text is stored locally for reproducible training. Source URLs and license metadata are retained for attribution. The collector accepts CC BY-SA 3.0/4.0 articles.</p></section>}

      {tab==='playground' && <section className="playground-grid"><div className="panel prompt-panel"><span className="eyebrow">INFERENCE WORKSPACE</span><h3>Test a trained model</h3><p className="muted">Generation uses a published checkpoint. It does not pause or modify the training worker.</p><label>MODEL VERSION</label><select value={selectedModel} onChange={e=>setSelectedModel(e.target.value)}><option value="">Choose a ready model</option>{readyModels.map(m=><option key={m.id} value={m.id}>{m.id}{m.parent_id?` · from ${m.parent_id}`:' · initial model'}</option>)}</select><label>STARTING PROMPT</label><textarea value={prompt} onChange={e=>setPrompt(e.target.value)} rows={5} placeholder="Enter at least 8 characters..."/><div className="control-row"><div><label>Temperature <b>{temperature.toFixed(1)}</b></label><input type="range" min="0" max="1.5" step="0.1" value={temperature} onChange={e=>setTemperature(Number(e.target.value))}/><small>0 = greedy · higher = more random</small></div><div><label>New characters</label><input type="number" min="1" max="500" value={maxTokens} onChange={e=>setMaxTokens(Number(e.target.value))}/></div></div><button className="primary-button" disabled={!selectedModel||!prompt.trim()||generating} onClick={()=>void runGenerate()}>{generating?'Generating…':'Generate text →'}</button>{readyModels.length===0&&<div className="hint">No trained model is available yet. The first one appears after the initial training cycle passes evaluation.</div>}</div><div className="panel output-panel"><div className="panel-heading"><div><span className="eyebrow">GENERATED OUTPUT</span><h3>Model response</h3></div><button className="text-button" onClick={()=>{void navigator.clipboard?.writeText(output)}} disabled={!output}>Copy</button></div>{output?<pre>{output}</pre>:<div className="output-empty"><div>✳</div><strong>Your generated text will appear here</strong><span>Choose a ready model and enter a prompt.</span></div>}</div></section>}
      <footer>Pattern Learner · Self-hosted research prototype <span>Updates automatically while the worker is running</span></footer>
    </main>
  </div>
}

function Stat({label,value,sub}:{label:string;value:string|number;sub:string}) { return <div className="stat-card"><span>{label}</span><strong>{value}</strong><small>{sub}</small></div> }
function Metric({label,value}:{label:string;value:string}) { return <div><small>{label}</small><strong>{value}</strong></div> }
function ModelTable({models,detailed=false}:{models:Model[];detailed?:boolean}) { return <div className="table-wrap"><table><thead><tr><th>MODEL</th><th>STATUS</th>{detailed&&<th>PARENT</th>}<th>VALIDATION LOSS</th><th>TEST LOSS</th><th>SOURCES</th>{detailed&&<th>CREATED</th>}</tr></thead><tbody>{models.map(m=><tr key={m.id}><td><strong>{m.id}</strong><small className="cell-sub">{m.parameter_count??'—'} parameters</small></td><td><span className={`status-pill ${m.status}`}>{m.status}</span>{m.status==='training'&&<small className="cell-sub">Epoch {m.epoch}/{m.total_epochs}</small>}{m.error&&<small className="error-text">{m.error}</small>}</td>{detailed&&<td>{m.parent_id||'—'}</td>}<td>{fmt(m.validation_loss)}</td><td>{fmt(m.test_loss)}</td><td>{m.source_count}</td>{detailed&&<td>{date(m.created_at)}</td>}</tr>)}</tbody></table>{models.length===0&&<Empty text="No models yet. The trainer will publish the first model after collecting enough articles."/>}</div> }
function EventList({events}:{events:EventItem[]}) { return <div className="event-list">{events.map((event,i)=><div className="event-item" key={`${event.id}-${i}`}><i className={event.type==='log'?'event-log':event.type==='model'?'event-model':'event-status'}></i><div><strong>{String(event.data.message||event.data.id||event.type)}</strong><small>{date(event.created_at)}</small></div></div>)}{events.length===0&&<Empty text="Waiting for worker events…"/>}</div> }
function Empty({text}:{text:string}) { return <div className="empty-state">{text}</div> }
