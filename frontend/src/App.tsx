import type {
  ChangeEvent,
  DragEvent,
  MouseEvent as ReactMouseEvent,
} from 'react'
import { useMemo, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeRaw from 'rehype-raw'
import './App.css'

type ParseResponse = {
  data: unknown
  visualizations?: string[]
  chunking_strategy?: string
}

type ChunkingStrategy = 'semantic' | 'late' | 'llm' | 'hierarchical' | 'agentic'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

const CHUNKING_OPTIONS: { value: ChunkingStrategy; label: string }[] = [
  { value: 'agentic', label: 'Agentic' },
  { value: 'semantic', label: 'Semantic' },
  { value: 'late', label: 'Late chunking' },
  { value: 'llm', label: 'LLM chunking' },
  { value: 'hierarchical', label: 'Hierarchical' },
]

const CHUNKING_DESCRIPTIONS: Record<ChunkingStrategy, string> = {
  semantic:
    'Keeps layout-aware chunks and preserves document structure. Good default that balances fidelity and chunk size.',
  late:
    'Embeds the full doc first, then chunks based on embedding similarity. Useful when you want semantic-first splits.',
  llm:
    'Uses LlamaParse with LLM-guided splitting to keep coherent passages. Great for narrative-heavy or long-form PDFs.',
  hierarchical:
    'Maintains parent/child nodes to mirror document hierarchy. Best when you need section context kept with children.',
  agentic:
    'Keeps layout-aware chunks and preserves document structure. Good default that balances fidelity and chunk size.',
}

function App() {
  const [file, setFile] = useState<File | null>(null)
  const [isDragging, setDragging] = useState(false)
  const [isUploaderOpen, setUploaderOpen] = useState(true)
  const [loading, setLoading] = useState(false)
  const [chatLoading, setChatLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [chunkingStrategy, setChunkingStrategy] =
    useState<ChunkingStrategy>('agentic')
  const [appliedChunkingStrategy, setAppliedChunkingStrategy] =
    useState<ChunkingStrategy>('agentic')
  const [chatError, setChatError] = useState<string | null>(null)
  const [parsedData, setParsedData] = useState<unknown | null>(null)
  const [panelMode, setPanelMode] = useState<'parse' | 'chat'>('parse')
  const [activeTab, setActiveTab] = useState<'json' | 'markdown'>('json')
  const [visualizations, setVisualizations] = useState<string[]>([])
  const [filteredVisualizations, setFilteredVisualizations] = useState<string[]>([])
  const [vizLoading, setVizLoading] = useState(false)
  const [showAllViz, setShowAllViz] = useState(true)
  const [splitPercent, setSplitPercent] = useState(50)
  const [chatQuestion, setChatQuestion] = useState('')
  const vizCardRefs = useRef<Record<number, HTMLDivElement | null>>({})
  type GroundingBox = { l: number; t: number; r: number; b: number }
  type GroundingRef = {
    page: number
    boxes: GroundingBox[]
    chunkId?: string | number
    chunkType?: string
    label: string
  }

  type ChatMessage = {
    question: string
    answer: string
    sources?: unknown[]
    groundingRefs?: GroundingRef[]
  }

  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([])
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const layoutRef = useRef<HTMLDivElement | null>(null)
  const [activeHighlights, setActiveHighlights] = useState<
    Record<number, { boxes: GroundingBox[]; pulse: number }>
  >({})

  const prettyJson = useMemo(
    () => (parsedData ? JSON.stringify(parsedData, null, 2) : ''),
    [parsedData]
  )

  const markdownText = useMemo(() => {
    if (!parsedData) return ''

    const collected: string[] = []
    const walk = (val: unknown) => {
      if (!val) return
      if (typeof val === 'string') return
      if (Array.isArray(val)) {
        val.forEach(walk)
        return
      }
      if (typeof val === 'object') {
        const obj = val as Record<string, unknown>
        if (typeof obj.markdown === 'string') {
          collected.push(obj.markdown)
        }
        if (Array.isArray(obj.documents)) {
          obj.documents.forEach(walk)
        }
        Object.values(obj).forEach(walk)
      }
    }

    walk(parsedData)
    return collected.filter(Boolean).join('\n\n---\n\n')
  }, [parsedData])

  const chunkingLabel = useMemo(() => {
    const labelMap: Record<string, string> = {
      semantic: 'Semantic',
      agentic: 'Semantic',
      late: 'Late chunking',
      llm: 'LLM chunking',
      hierarchical: 'Hierarchical',
    }
    return labelMap[appliedChunkingStrategy] ?? appliedChunkingStrategy
  }, [appliedChunkingStrategy])

  const handleFileSelect = (selected: File | null) => {
    if (!selected) return null
    const isPdf =
      selected.type === 'application/pdf' ||
      selected.name.toLowerCase().endsWith('.pdf')
    if (!isPdf) {
      setError('Only PDF files are supported.')
      setFile(null)
      return null
    }
    setError(null)
    setFile(selected)
    return selected
  }

  const onFileInputChange = (event: ChangeEvent<HTMLInputElement>) => {
    const selected = event.target.files?.[0] ?? null
    const valid = handleFileSelect(selected)
    if (valid) {
      submit(valid)
    }
    event.target.value = ''
  }

  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    event.stopPropagation()
    setDragging(false)
    const dropped = event.dataTransfer.files?.[0]
    const valid = handleFileSelect(dropped ?? null)
    if (valid) {
      submit(valid)
    }
  }

  const onDragOver = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    event.stopPropagation()
    setDragging(true)
  }

  const onDragLeave = () => setDragging(false)

  const submit = async (targetFile: File | null = file) => {
    const fileToUpload = targetFile ?? file
    if (!fileToUpload) {
      setError('Choose a PDF to parse.')
      setUploaderOpen(true)
      return
    }
    setLoading(true)
    setError(null)
    setParsedData(null)
    setVisualizations([])
    setFilteredVisualizations([])
    setVizLoading(false)
    setShowAllViz(true)
    setChatMessages([])
    setChatQuestion('')
    setActiveHighlights({})
    setFile(fileToUpload)

    const effectiveStrategy =
      chunkingStrategy === 'agentic' ? 'semantic' : chunkingStrategy

    const form = new FormData()
    form.append('file', fileToUpload)
    form.append('chunking_strategy', effectiveStrategy)

    try {
      const res = await fetch(`${API_BASE}/api/parse`, {
        method: 'POST',
        body: form,
      })

      if (!res.ok) {
        const detail = await res.json().catch(() => null)
        const message =
          detail?.detail || `Request failed with status ${res.status}`
        throw new Error(message)
      }

      const body = (await res.json()) as ParseResponse
      setParsedData(body?.data ?? body)
      setVisualizations(body?.visualizations ?? [])
      const returnedStrategy = body?.chunking_strategy ?? chunkingStrategy
      setAppliedChunkingStrategy(returnedStrategy as ChunkingStrategy)
      setUploaderOpen(false)
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Unexpected error occurred.'
      setError(message)
    } finally {
      setLoading(false)
    }
  }

  async function fetchFilteredVisualizations(chunkIds: (string | number)[]) {
    if (!chunkIds || chunkIds.length === 0) {
      setFilteredVisualizations([])
      setShowAllViz(true)
      return
    }
    setVizLoading(true)
    try {
      const res = await fetch(`${API_BASE}/api/chat/visualizations`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ chunk_ids: chunkIds.map((cid) => String(cid)) }),
      })
      if (!res.ok) {
        const detail = await res.json().catch(() => null)
        const message =
          detail?.detail || `Visualization request failed with status ${res.status}`
        throw new Error(message)
      }
      const body = (await res.json()) as { filtered_visualizations?: string[] }
      const filtered = body.filtered_visualizations ?? []
      setFilteredVisualizations(filtered)
      if (filtered.length > 0) {
        setShowAllViz(false)
      }
    } catch (err) {
      console.error('Filtered visualization fetch failed', err)
      setFilteredVisualizations([])
    } finally {
      setVizLoading(false)
    }
  }

  const submitChat = async () => {
    if (!parsedData) {
      setChatError('Parse a PDF first.')
      return
    }
    if (!chatQuestion.trim()) {
      setChatError('Enter a question.')
      return
    }

    setChatError(null)
    setChatLoading(true)
    try {
      const res = await fetch(`${API_BASE}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: chatQuestion,
          parsed: parsedData,
        }),
      })
      if (!res.ok) {
        const detail = await res.json().catch(() => null)
        const message =
          detail?.detail || `Chat failed with status ${res.status}`
        throw new Error(message)
      }
      const body = (await res.json()) as {
        message: string
        sources?: unknown[]
        filtered_visualizations?: string[]
        chunk_ids?: (string | number)[]
      }
      // Display-only: try to extract answer field from JSON, else show raw
      const displayAnswer = extractAnswerFromMessage(body.message)
      const usedChunkIds = extractChunkIdsUsed(body.message)
      const chunkIdsFromResponse = Array.isArray(body.chunk_ids)
        ? body.chunk_ids.map((cid) => String(cid))
        : []
      const effectiveChunkIds = new Set<string>(chunkIdsFromResponse)
      usedChunkIds.forEach((cid) => effectiveChunkIds.add(cid))

      const filteredSources =
        effectiveChunkIds.size > 0 && body.sources
          ? body.sources.filter((src) => {
              if (!src || typeof src !== 'object') return false
              const cid = (src as Record<string, unknown>).chunk_id
              return cid !== undefined && cid !== null && effectiveChunkIds.has(String(cid))
            })
          : body.sources ?? []
      const groundingRefs = filteredSources.length > 0 ? parseGroundingRefs(filteredSources) : []
      setChatMessages([
        {
          question: chatQuestion,
          answer: displayAnswer,
          sources: filteredSources,
          groundingRefs,
        },
      ])
      const filteredFromBody = body.filtered_visualizations ?? []
      if (filteredFromBody.length > 0) {
        setFilteredVisualizations(filteredFromBody)
        setShowAllViz(false)
      } else {
        const chunkIdsArray = effectiveChunkIds.size > 0 ? Array.from(effectiveChunkIds) : []
        setFilteredVisualizations([])
        if (chunkIdsArray.length === 0) {
          setShowAllViz(true)
        } else {
          setShowAllViz(false)
          void fetchFilteredVisualizations(chunkIdsArray)
        }
      }
      setChatQuestion('')
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Unexpected chat error.'
      setChatError(message)
    } finally {
      setChatLoading(false)
    }
  }

  const copyCurrent = async () => {
    const text =
      activeTab === 'json' ? prettyJson : markdownText || 'No markdown found.'
    if (!text) return
    try {
      await navigator.clipboard.writeText(text)
    } catch (err) {
      console.error('Copy failed', err)
    }
  }

  const getImageUrl = (src: string) =>
    src.startsWith('http') ? src : `${API_BASE}${src}`

  const getPageIndexFromSrc = (src: string): number | null => {
    const match = src.match(/page_(\d+)/)
    return match ? Number(match[1]) : null
  }

  const extractChunkIdsUsed = (msg: string): Set<string> => {
    const ids = new Set<string>()
    if (!msg) return ids

    const tryParse = (text: string) => {
      try {
        const parsed = JSON.parse(text)
        const chunks = Array.isArray(parsed?.chunks_used) ? parsed.chunks_used : []
        chunks.forEach((item: unknown) => {
          if (item && typeof item === 'object' && 'chunk_id' in item) {
            const cid = (item as Record<string, unknown>).chunk_id
            if (cid !== undefined && cid !== null) ids.add(String(cid))
          } else if (typeof item === 'string') {
            ids.add(item)
          }
        })
      } catch {
        /* ignore */
      }
    }

    if (msg.includes('```json')) {
      const split = msg.split('```json')
      const after = split[1] ?? ''
      const endIdx = after.indexOf('```')
      const candidate = endIdx >= 0 ? after.slice(0, endIdx).trim() : after.trim()
      tryParse(candidate)
    }
    tryParse(msg.trim())
    return ids
  }

  const parseGroundingRefs = (sourcesRaw: unknown[]): GroundingRef[] => {
    const pageMap: Record<number, GroundingRef> = {}

    sourcesRaw.forEach((src) => {
      if (!src || typeof src !== 'object') return
      const meta = src as Record<string, unknown>
      const chunkType = typeof meta.chunk_type === 'string' ? meta.chunk_type : undefined
      const chunkId = meta.chunk_id as string | number | undefined
      const defaultPage =
        typeof meta.page_number === 'number' && !Number.isNaN(meta.page_number)
          ? meta.page_number
          : undefined
      const groundingRaw = meta.grounding_info
      let groundingList: unknown[] = []
      if (typeof groundingRaw === 'string') {
        try {
          const parsed = JSON.parse(groundingRaw)
          if (Array.isArray(parsed)) groundingList = parsed
        } catch {
          /* ignore */
        }
      } else if (Array.isArray(groundingRaw)) {
        groundingList = groundingRaw
      } else if (groundingRaw && typeof groundingRaw === 'object') {
        groundingList = [groundingRaw]
      }

      groundingList.forEach((g) => {
        if (!g || typeof g !== 'object') return
        const gObj = g as Record<string, any>
        const page =
          typeof gObj.page === 'number' && !Number.isNaN(gObj.page) ? gObj.page : defaultPage
        if (page === undefined || page === null) return

        const boxCandidate = gObj.box ?? gObj
        const l = Number(boxCandidate.l ?? boxCandidate.left)
        const t = Number(boxCandidate.t ?? boxCandidate.top)
        const r = Number(boxCandidate.r ?? boxCandidate.right)
        const b = Number(boxCandidate.b ?? boxCandidate.bottom)
        if ([l, t, r, b].some((v) => Number.isNaN(v))) return

        const cell =
          typeof gObj.cell === 'string'
            ? gObj.cell
            : typeof gObj.cell_id === 'string'
              ? gObj.cell_id
              : undefined
        const labelParts = [`Page ${page + 1}`]
        if (chunkType) labelParts.push(chunkType)
        if (cell) labelParts.push(cell)
        const label = labelParts.join('. ')

        if (!pageMap[page]) {
          pageMap[page] = {
            page,
            boxes: [],
            chunkId,
            chunkType,
            label,
          }
        }
        pageMap[page].boxes.push({ l, t, r, b })
      })
    })

    return Object.values(pageMap)
  }

  const triggerHighlight = (ref: GroundingRef) => {
    if (!ref.boxes.length || ref.page === undefined || ref.page === null) return
    const card = vizCardRefs.current[ref.page]
    if (card?.scrollIntoView) {
      card.scrollIntoView({ behavior: 'smooth', block: 'center' })
    }
    setActiveHighlights((prev) => {
      const pulse = (prev[ref.page]?.pulse ?? 0) + 1
      const next = { ...prev, [ref.page]: { boxes: ref.boxes, pulse } }
      setTimeout(() => {
        setActiveHighlights((current) => {
          const latest = current[ref.page]
          if (!latest || latest.pulse !== pulse) return current
          const copy = { ...current }
          delete copy[ref.page]
          return copy
        })
      }, 1200)
      return next
    })
  }

  const onDividerMouseDown = (event: ReactMouseEvent<HTMLDivElement>) => {
    event.preventDefault()
    const startX = event.clientX
    const startPercent = splitPercent
    const layout = layoutRef.current
    if (!layout) return
    const rect = layout.getBoundingClientRect()

    const onMove = (moveEvent: globalThis.MouseEvent) => {
      const delta = moveEvent.clientX - startX
      const deltaPercent = (delta / rect.width) * 100
      const next = Math.min(70, Math.max(30, startPercent + deltaPercent))
      setSplitPercent(next)
    }

    const onUp = () => {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
    }

    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
  }

  const pageClassName = `page ${isUploaderOpen ? 'modal-open' : ''}`
  const displayedVisualizations = showAllViz ? visualizations : filteredVisualizations
  const totalVisualizations = visualizations.length
  const vizCountLabel =
    totalVisualizations === 0
      ? '—'
      : showAllViz || filteredVisualizations.length === 0
        ? `${totalVisualizations} page(s)`
        : `${displayedVisualizations.length} / ${totalVisualizations} page(s)`

  const extractAnswerFromMessage = (msg: string) => {
    if (!msg) return ''
    // try fenced JSON first
    const fenceSplit = msg.split('```json')
    const tryParse = (text: string) => {
      try {
        const parsed = JSON.parse(text)
        if (parsed && typeof parsed === 'object' && 'answer' in parsed) {
          const ans = (parsed as Record<string, unknown>).answer
          if (typeof ans === 'string') return ans
        }
      } catch {
        /* ignore */
      }
      return ''
    }
    if (fenceSplit.length > 1) {
      const after = fenceSplit[1]
      const endIdx = after.indexOf('```')
      const candidate = endIdx >= 0 ? after.slice(0, endIdx).trim() : after.trim()
      const parsed = tryParse(candidate)
      if (parsed) return parsed
    }
    const parsedRaw = tryParse(msg.trim())
    if (parsedRaw) return parsedRaw
    return msg
  }

  return (
    <div className={pageClassName}>
      <header className="hero">
        <div>
          <p className="eyebrow">Grounded QA — doc ingest</p>
          <h1>Knowledge extraction from PDFs</h1>
          <p className="sub">
          </p>
        </div>
      </header>

      {error && <div className="alert error">{error}</div>}
      {loading && (
        <div className="loading banner">
          <div className="spinner" aria-hidden />
          <p>Parsing PDF…</p>
        </div>
      )}

      <input
        ref={fileInputRef}
        type="file"
        accept="application/pdf"
        onChange={onFileInputChange}
        hidden
      />

      <div className="app-shell">
        <aside className="sidebar">
          <button
            className="icon-button"
            onClick={() => setUploaderOpen(true)}
            title="Upload PDF"
          >
            <svg viewBox="0 0 24 24" aria-hidden>
              <path
                d="M12 16V7m0 0l-3 3m3-3l3 3M5 17a4 4 0 010-8 5 5 0 019.5-2.5A4 4 0 1118 17H5z"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.7"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </button>
        </aside>
        <div className="layout split" ref={layoutRef}>
          <section
            className="panel column split-panel"
            style={{ flexBasis: `${splitPercent}%` }}
          >
            <div className="panel-header">
              <div>
                <p className="eyebrow">Visual previews</p>
                <h3>Chunk overlays</h3>
                {file && <p className="file-chip">{file.name}</p>}
              </div>
              {filteredVisualizations.length > 0 && (
                <label className="viz-toggle">
                  <input
                    type="checkbox"
                    checked={showAllViz}
                    onChange={(e) => setShowAllViz(e.target.checked)}
                  />
                  Show all page overlays
                </label>
              )}
              <p className="viz-count">{vizCountLabel}</p>
            </div>
            <div className="viz-board">
              {vizLoading ? (
                <p className="placeholder">Fetching referenced overlays…</p>
              ) : displayedVisualizations.length > 0 ? (
                <div className="viz-grid">
                  {displayedVisualizations.map((src) => {
                    const pageIdx = getPageIndexFromSrc(src)
                    const highlight = pageIdx !== null ? activeHighlights[pageIdx] : undefined
                    return (
                      <div
                        className={`viz-card ${highlight ? 'has-highlight' : ''}`}
                        key={src}
                        ref={(el) => {
                          if (pageIdx !== null) {
                            vizCardRefs.current[pageIdx] = el
                          }
                        }}
                      >
                        <img src={getImageUrl(src)} alt="Visualization" />
                        {highlight && (
                          <div className="viz-overlay" key={highlight.pulse}>
                            {highlight.boxes.map((box, idx) => (
                              <span
                                key={`${highlight.pulse}-${idx}`}
                                className="viz-highlight"
                                style={{
                                  left: `${box.l * 100}%`,
                                  top: `${box.t * 100}%`,
                                  width: `${(box.r - box.l) * 100}%`,
                                  height: `${(box.b - box.t) * 100}%`,
                                }}
                              />
                            ))}
                          </div>
                        )}
                      </div>
                    )
                  })}
                </div>
              ) : (
                <p className="placeholder">
                  {showAllViz
                    ? 'Upload a PDF to see page overlays.'
                    : 'Ask a question to reveal the referenced overlays.'}
                </p>
              )}
            </div>
          </section>

          <div
            className="splitter"
            onMouseDown={onDividerMouseDown}
            role="separator"
            aria-label="Resize panels"
          />

          <section
            className="panel column split-panel parser-panel"
            style={{ flexBasis: `${100 - splitPercent}%` }}
          >
            <div className="mode-tabs">
              <button
                className={`tab ${panelMode === 'parse' ? 'active' : ''}`}
                onClick={() => setPanelMode('parse')}
              >
                Parse
              </button>
              <button
                className={`tab ${panelMode === 'chat' ? 'active' : ''}`}
                onClick={() => setPanelMode('chat')}
                disabled={!parsedData}
              >
                Chat
              </button>
            </div>

            {panelMode === 'parse' ? (
              <>
                <div className="panel-header">
                  <div>
                    <p className="eyebrow">Parse output</p>
                    <h3>Structured response</h3>
                    <p className="eyebrow muted">
                      Chunking strategy: {chunkingLabel}
                    </p>
                  </div>
                  <div className="viewer-actions">
                    <div className="tabs">
                      <button
                        className={`tab ${activeTab === 'json' ? 'active' : ''}`}
                        onClick={() => setActiveTab('json')}
                        disabled={!parsedData}
                      >
                        JSON
                      </button>
                      <button
                        className={`tab ${
                          activeTab === 'markdown' ? 'active' : ''
                        }`}
                        onClick={() => setActiveTab('markdown')}
                        disabled={!parsedData}
                      >
                        Markdown
                      </button>
                    </div>
                    <button
                      className="secondary"
                      onClick={copyCurrent}
                      disabled={!parsedData}
                    >
                      {activeTab === 'json' ? 'Copy JSON' : 'Copy Markdown'}
                    </button>
                  </div>
                </div>
                <div
                  className={`json-viewer ${
                    activeTab === 'markdown' ? 'markdown-mode' : ''
                  }`}
                >
                  {activeTab === 'json' ? (
                    prettyJson ? (
                      <pre>{prettyJson}</pre>
                    ) : (
                      <p className="placeholder">Upload a PDF to see the parsed payload.</p>
                    )
                  ) : markdownText ? (
                    <div className="markdown-body">
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        rehypePlugins={[rehypeRaw]}
                      >
                        {markdownText}
                      </ReactMarkdown>
                    </div>
                  ) : (
                    <p className="placeholder">No markdown was returned for this parse.</p>
                  )}
                </div>
              </>
            ) : (
              <div className="chat-panel">
                <div className="panel-header">
                  <div>
                    <p className="eyebrow">Grounded answers</p>
                    <h3>Ask about this PDF</h3>
                  </div>
                </div>
                {chatError && <div className="alert error">{chatError}</div>}
                <div className="chat-box">
                  <textarea
                    className="chat-input"
                    placeholder="Ask something about the parsed PDF…"
                    value={chatQuestion}
                    onChange={(e) => setChatQuestion(e.target.value)}
                    disabled={chatLoading}
                  />
                  <button
                    className="primary"
                    onClick={submitChat}
                    disabled={chatLoading}
                  >
                    {chatLoading ? 'Asking…' : 'Ask'}
                  </button>
                </div>
                <div className="chat-messages">
                  {chatMessages.length === 0 ? (
                    <p className="placeholder">Ask to see a grounded answer.</p>
                  ) : (
                    chatMessages.map((msg, idx) => (
                      <div className="chat-message" key={idx}>
                        <p className="chat-question">Q: {msg.question}</p>
                        <pre className="chat-answer">{msg.answer}</pre>
                        {msg.groundingRefs && msg.groundingRefs.length > 0 && (
                          <div className="sources-list">
                            <p className="eyebrow muted">Visual reference for this answer</p>
                            <div className="source-pills">
                              {msg.groundingRefs.map((ref, rIdx) => (
                                <button
                                  key={`${ref.page}-${rIdx}`}
                                  className="source-pill"
                                  type="button"
                                  onClick={() => triggerHighlight(ref)}
                                  title="Highlight on the visualization"
                                >
                                  <span>{ref.label}</span>
                                  <span aria-hidden>→</span>
                                </button>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    ))
                  )}
                </div>
              </div>
            )}
          </section>
        </div>
      </div>

      {isUploaderOpen && (
        <div
          className="upload-modal"
          role="dialog"
          aria-modal="true"
          onClick={() => {
            setDragging(false)
            setUploaderOpen(false)
          }}
        >
          <div
            className="upload-card"
            onClick={(event) => {
              event.stopPropagation()
            }}
          >
            {loading ? (
              <div className="upload-loading">
                <p className="modal-title">Processing PDF</p>
                <p className="modal-sub">Hold tight while we parse and prep your overlays.</p>
                <div className="loading-steps">
                  <div className="loading-step done">
                    <span className="step-dot" aria-hidden />
                    <div className="step-text">
                      <p className="step-title">Upload received</p>
                      <p className="step-sub">File transferred securely.</p>
                    </div>
                    <span className="step-check" aria-hidden>✓</span>
                  </div>
                  <div className="loading-step active">
                    <span className="step-dot" aria-hidden />
                    <div className="step-text">
                      <p className="step-title">Parsing & chunking</p>
                      <p className="step-sub">Extracting text and building chunks.</p>
                    </div>
                    <div className="mini-spinner" aria-hidden />
                  </div>
                  <div className="loading-step pending">
                    <span className="step-dot" aria-hidden />
                    <div className="step-text">
                      <p className="step-title">Visual overlays</p>
                      <p className="step-sub">Preparing page previews.</p>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <>
                <p className="modal-title">Add a PDF</p>
                <p className="modal-sub">Drop a PDF to parse instantly, or click to browse.</p>
                <div className="option-row column">
                  <label className="option-label" htmlFor="chunking-select">
                    Chunking strategy
                  </label>
                  <select
                    id="chunking-select"
                    value={chunkingStrategy}
                    onChange={(e) =>
                      setChunkingStrategy(e.target.value as ChunkingStrategy)
                    }
                    className="select-control"
                    disabled={loading}
                  >
                    {CHUNKING_OPTIONS.map((opt) => (
                      <option key={opt.value} value={opt.value}>
                        {opt.label}
                      </option>
                    ))}
                  </select>
                  <p className="option-help">
                    {CHUNKING_DESCRIPTIONS[chunkingStrategy]}
                  </p>
                </div>
                <div
                  className={`dropzone ${isDragging ? 'dragging' : ''} ${loading ? 'loading' : ''}`}
                  onDragOver={onDragOver}
                  onDragLeave={onDragLeave}
                  onDrop={onDrop}
                  onClick={() => fileInputRef.current?.click()}
                >
                  <p className="drop-title">Drop your PDF here</p>
                  <p className="drop-sub">We’ll start parsing the moment it lands.</p>
                  {file ? (
                    <p className="file-name">Selected: {file.name}</p>
                  ) : (
                    <p className="file-name muted">No file selected yet</p>
                  )}
                </div>
                <div className="modal-actions">
                  <button
                    className="secondary"
                    onClick={() => {
                      setDragging(false)
                      setUploaderOpen(false)
                    }}
                    disabled={loading}
                  >
                    Close
                  </button>
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

export default App
