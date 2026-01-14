import type { ChangeEvent, DragEvent, MouseEvent } from 'react'
import { useMemo, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeRaw from 'rehype-raw'
import './App.css'

type ParseResponse = {
  data: unknown
  visualizations?: string[]
}

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

function App() {
  const [file, setFile] = useState<File | null>(null)
  const [isDragging, setDragging] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [parsedData, setParsedData] = useState<unknown | null>(null)
  const [activeTab, setActiveTab] = useState<'json' | 'markdown'>('json')
  const [visualizations, setVisualizations] = useState<string[]>([])
  const [splitPercent, setSplitPercent] = useState(50)
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

  const handleFileSelect = (selected: File | null) => {
    if (!selected) return
    if (selected.type !== 'application/pdf') {
      setError('Only PDF files are supported.')
      setFile(null)
      return
    }
    setError(null)
    setFile(selected)
  }

  const onFileInputChange = (event: ChangeEvent<HTMLInputElement>) => {
    const selected = event.target.files?.[0] ?? null
    handleFileSelect(selected)
  }

  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setDragging(false)
    const dropped = event.dataTransfer.files?.[0]
    handleFileSelect(dropped ?? null)
  }

  const onDragOver = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setDragging(true)
  }

  const onDragLeave = () => setDragging(false)

  const submit = async () => {
    if (!file) {
      setError('Choose a PDF to parse.')
      return
    }
    setLoading(true)
    setError(null)
    setParsedData(null)
    setVisualizations([])

    const form = new FormData()
    form.append('file', file)

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
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Unexpected error occurred.'
      setError(message)
    } finally {
      setLoading(false)
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

  const onDividerMouseDown = (event: MouseEvent<HTMLDivElement>) => {
    event.preventDefault()
    const startX = event.clientX
    const startPercent = splitPercent
    const layout = layoutRef.current
    if (!layout) return
    const rect = layout.getBoundingClientRect()

    const onMove = (moveEvent: MouseEvent) => {
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

  return (
    <div className="page">
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

      <div className="app-shell">
        <aside className="sidebar">
          <input
            ref={fileInputRef}
            type="file"
            accept="application/pdf"
            onChange={onFileInputChange}
            hidden
          />
          <button
            className="icon-button"
            onClick={() => fileInputRef.current?.click()}
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
          <button
            className="icon-button"
            onClick={submit}
            disabled={loading}
            title="Parse"
          >
            <svg viewBox="0 0 24 24" aria-hidden>
              <path
                d="M5 12h14M12 5v14"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.7"
                strokeLinecap="round"
              />
            </svg>
          </button>
          <button
            className="icon-button"
            onClick={() => setFile(null)}
            title="Clear"
          >
            <svg viewBox="0 0 24 24" aria-hidden>
              <path
                d="M6 6l12 12M18 6l-12 12"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.7"
                strokeLinecap="round"
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
              <p className="viz-count">
                {visualizations.length > 0 ? `${visualizations.length} page(s)` : '—'}
              </p>
            </div>
            <div className="viz-board">
              {visualizations.length > 0 ? (
                <div className="viz-grid">
                  {visualizations.map((src) => (
                    <div className="viz-card" key={src}>
                      <img src={getImageUrl(src)} alt="Visualization" />
                    </div>
                  ))}
                </div>
              ) : (
                <p className="placeholder">Upload to view page visualizations.</p>
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
            className="panel column split-panel"
            style={{ flexBasis: `${100 - splitPercent}%` }}
          >
            <div className="panel-header">
              <div>
                <p className="eyebrow">Parser output</p>
                <h3>Pretty-printed response</h3>
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
                    className={`tab ${activeTab === 'markdown' ? 'active' : ''}`}
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
          </section>
        </div>
      </div>
    </div>
  )
}

export default App
