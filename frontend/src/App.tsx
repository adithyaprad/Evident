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
  { value: 'semantic', label: 'Semantic (default)' },
  { value: 'late', label: 'Late chunking (Jina)' },
  { value: 'llm', label: 'LLM chunking (LlamaParse)' },
  { value: 'hierarchical', label: 'Hierarchical (LlamaParse)' },
]

function App() {
  const [file, setFile] = useState<File | null>(null)
  const [isDragging, setDragging] = useState(false)
  const [isUploaderOpen, setUploaderOpen] = useState(false)
  const [loading, setLoading] = useState(false)
  const [chatLoading, setChatLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [chunkingStrategy, setChunkingStrategy] =
    useState<ChunkingStrategy>('semantic')
  const [appliedChunkingStrategy, setAppliedChunkingStrategy] =
    useState<ChunkingStrategy>('semantic')
  const [chatError, setChatError] = useState<string | null>(null)
  const [parsedData, setParsedData] = useState<unknown | null>(null)
  const [panelMode, setPanelMode] = useState<'parse' | 'chat'>('parse')
  const [activeTab, setActiveTab] = useState<'json' | 'markdown'>('json')
  const [visualizations, setVisualizations] = useState<string[]>([])
  const [filteredVisualizations, setFilteredVisualizations] = useState<string[]>([])
  const [showAllViz, setShowAllViz] = useState(true)
  const [splitPercent, setSplitPercent] = useState(50)
  const [chatQuestion, setChatQuestion] = useState('')
  const [chatMessages, setChatMessages] = useState<
    { question: string; answer: string; sources?: unknown[] }[]
  >([])
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const layoutRef = useRef<HTMLDivElement | null>(null)

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
      semantic: 'Semantic (default)',
      agentic: 'Semantic (default)',
      late: 'Late chunking (Jina)',
      llm: 'LLM chunking (LlamaParse)',
      hierarchical: 'Hierarchical (LlamaParse)',
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
      setUploaderOpen(false)
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
      setUploaderOpen(false)
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
      return
    }
    setLoading(true)
    setError(null)
    setParsedData(null)
    setVisualizations([])
    setFile(fileToUpload)

    const form = new FormData()
    form.append('file', fileToUpload)
    form.append('chunking_strategy', chunkingStrategy)

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
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Unexpected error occurred.'
      setError(message)
    } finally {
      setLoading(false)
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
      }
      // Display-only: try to extract answer field from JSON, else show raw
      const displayAnswer = extractAnswerFromMessage(body.message)
      setChatMessages([
        { question: chatQuestion, answer: displayAnswer, sources: body.sources },
      ])
      const filtered = body.filtered_visualizations ?? []
      setFilteredVisualizations(filtered)
      if (filtered.length > 0) {
        setShowAllViz(false)
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
          <p className="eyebrow">Grounded QA — raw ingest</p>
          <h1>Upload a PDF and inspect the parser output</h1>
          <p className="sub">
            Uses the existing <code>agentic_doc.parse</code> flow; JSON,
            markdown, and visualization previews.
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
                <p className="eyebrow">Visualizations</p>
                <h3>Page overlays</h3>
                {file && <p className="file-chip">{file.name}</p>}
              </div>
              {filteredVisualizations.length > 0 && (
                <label className="viz-toggle">
                  <input
                    type="checkbox"
                    checked={showAllViz}
                    onChange={(e) => setShowAllViz(e.target.checked)}
                  />
                  Show all chunk visualizations
                </label>
              )}
              <p className="viz-count">
                {visualizations.length > 0 ? `${visualizations.length} page(s)` : '—'}
              </p>
            </div>
            <div className="viz-board">
              {(showAllViz ? visualizations : filteredVisualizations).length > 0 ? (
                <div className="viz-grid">
                  {(showAllViz ? visualizations : filteredVisualizations).map((src) => (
                    <div className="viz-card" key={src}>
                      <img src={getImageUrl(src)} alt="Visualization" />
                    </div>
                  ))}
                </div>
              ) : (
                <p className="placeholder">
                  {showAllViz
                    ? 'Upload to view page visualizations.'
                    : 'Ask a question to view referenced chunk visualizations.'}
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
                    <p className="eyebrow">Parser output</p>
                    <h3>Pretty-printed response</h3>
                    <p className="eyebrow muted">
                      Chunking: {chunkingLabel}
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
                      <p className="placeholder">
                        Upload a PDF to view the raw parse output.
                      </p>
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
                    <p className="placeholder">
                      No markdown found in the parser response.
                    </p>
                  )}
                </div>
              </>
            ) : (
              <div className="chat-panel">
                <div className="panel-header">
                  <div>
                    <p className="eyebrow">RAG chat</p>
                    <h3>Ask questions about this PDF</h3>
                  </div>
                </div>
                {chatError && <div className="alert error">{chatError}</div>}
                <div className="chat-box">
                  <textarea
                    className="chat-input"
                    placeholder="Ask a question about the parsed document..."
                    value={chatQuestion}
                    onChange={(e) => setChatQuestion(e.target.value)}
                    disabled={chatLoading}
                  />
                  <button
                    className="primary"
                    onClick={submitChat}
                    disabled={chatLoading}
                  >
                    {chatLoading ? 'Asking…' : 'Send'}
                  </button>
                </div>
                <div className="chat-messages">
                  {chatMessages.length === 0 ? (
                    <p className="placeholder">
                      Ask a question to see the model response.
                    </p>
                  ) : (
                    chatMessages.map((msg, idx) => (
                      <div className="chat-message" key={idx}>
                        <p className="chat-question">Q: {msg.question}</p>
                        <pre className="chat-answer">{msg.answer}</pre>
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
            <p className="modal-title">Upload a PDF</p>
            <p className="modal-sub">
              Drag and drop a PDF to start parsing automatically, or click to browse.
            </p>
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
              >
                {CHUNKING_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
              <p className="option-help">
                Semantic keeps layout-aware chunks. Late embeds full doc first.
                LLM uses LlamaParse + LLM-guided splits. Hierarchical keeps
                parent/child nodes.
              </p>
            </div>
            <div
              className={`dropzone ${isDragging ? 'dragging' : ''}`}
              onDragOver={onDragOver}
              onDragLeave={onDragLeave}
              onDrop={onDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <p className="drop-title">Drop your PDF here</p>
              <p className="drop-sub">
                We will blur the background and start parsing right away.
              </p>
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
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
