import type { ChangeEvent, DragEvent } from 'react'
import { useMemo, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeRaw from 'rehype-raw'
import './App.css'

type ParseResponse = {
  data: unknown
}

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

function App() {
  const [file, setFile] = useState<File | null>(null)
  const [isDragging, setDragging] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [parsedData, setParsedData] = useState<unknown | null>(null)
  const [activeTab, setActiveTab] = useState<'json' | 'markdown'>('json')
  const fileInputRef = useRef<HTMLInputElement | null>(null)

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

  return (
    <div className="page">
      <header className="hero">
        <div>
          <p className="eyebrow">Grounded QA — raw ingest</p>
          <h1>Upload a PDF and inspect the parser output</h1>
          <p className="sub">
            Uses the existing <code>agentic_doc.parse</code> flow; JSON is
            pretty-printed as-is for quick inspection.
          </p>
        </div>
      </header>

      <div className="grid">
        <section className="panel upload-panel">
          <div
            className={`dropzone ${isDragging ? 'dragging' : ''}`}
            onDrop={onDrop}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="application/pdf"
              onChange={onFileInputChange}
              hidden
            />
            <div className="dropzone-text">
              <p className="drop-title">Drop a PDF here or click to browse</p>
              <p className="drop-sub">Max ~25 MB. One file at a time.</p>
              {file ? (
                <p className="file-name">Selected: {file.name}</p>
              ) : (
                <p className="file-name muted">No file chosen</p>
              )}
            </div>
          </div>

          <div className="actions">
            <button className="secondary" onClick={() => setFile(null)}>
              Clear
            </button>
            <button className="primary" onClick={submit} disabled={loading}>
              {loading ? 'Parsing…' : 'Upload & Parse'}
            </button>
          </div>

          {error && <div className="alert error">{error}</div>}
          {!error && (
            <div className="hint">
              The backend endpoint: <code>{API_BASE}/api/parse</code>
            </div>
          )}

          {loading && (
            <div className="loading">
              <div className="spinner" aria-hidden />
              <p>Parsing PDF…</p>
            </div>
          )}
        </section>

        <section className="panel viewer">
          <div className="viewer-header">
            <div>
              <p className="eyebrow">Raw parser JSON</p>
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
  )
}

export default App
