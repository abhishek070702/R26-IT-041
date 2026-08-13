import { useEffect, useMemo, useRef, useState } from 'react'
import axios from 'axios'
import './App.css'

const API_BASE = 'http://127.0.0.1:8000'

const MODES = [
  {
    id: 'page',
    title: 'Read this page',
    hint: 'Point the wearable at a novel, magazine, newspaper, report, or letter.',
    api: 'analyze',
  },
  {
    id: 'image',
    title: 'Describe this image',
    hint: 'Use when the page has an illustration or a selected newspaper article photo.',
    api: 'describe',
  },
]

const CONTEXT_BY_HINT = [
  { value: 'novel_cover', label: 'Novel cover' },
  { value: 'novel_page', label: 'Story page illustration' },
  { value: 'newspaper_article_image', label: 'Selected article photo' },
  { value: 'newspaper_page', label: 'Newspaper page' },
  { value: 'general', label: 'General image' },
]

function speakText(text, { onStart, onEnd } = {}) {
  if (!text || !window.speechSynthesis) {
    onEnd?.()
    return
  }
  window.speechSynthesis.cancel()
  const utterance = new SpeechSynthesisUtterance(text)
  utterance.rate = 0.92
  utterance.pitch = 1
  utterance.onstart = () => onStart?.()
  utterance.onend = () => onEnd?.()
  utterance.onerror = () => onEnd?.()
  window.speechSynthesis.speak(utterance)
}

function stopSpeaking() {
  if (window.speechSynthesis) window.speechSynthesis.cancel()
}

function formatConfidence(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return null
  const num = Number(value)
  const pct = num <= 1 ? num * 100 : num
  return `${pct.toFixed(0)}%`
}

function joinDescriptions(descriptions) {
  if (!Array.isArray(descriptions) || descriptions.length === 0) return ''
  return descriptions.filter(Boolean).join(' ')
}

function backendErrorMessage(error) {
  if (error?.code === 'ERR_NETWORK' || error?.message === 'Network Error') {
    return 'Wearable offline. Start the backend at http://127.0.0.1:8000'
  }
  if (error?.response?.data?.detail) return String(error.response.data.detail)
  return error?.message || 'Something went wrong. Please try again.'
}

function buildAnalyzeSpeech(result) {
  if (!result) return ''
  const type = result.document_type || 'an unknown document'
  const title = result.title || ''
  const description = joinDescriptions(result.image_descriptions)
  const conf = formatConfidence(result.confidence)

  const parts = [`I detected a ${type}.`]
  if (conf) parts.push(`Confidence ${conf}.`)
  if (title && !String(title).toLowerCase().includes('handled by ocr')) {
    if (String(title).toLowerCase().includes('not clearly')) {
      parts.push('The title is not clearly detected.')
    } else if (String(title).toLowerCase().includes('newspaper name not')) {
      parts.push('The newspaper name is not clearly detected.')
    } else {
      parts.push(`The title is ${title}.`)
    }
  }
  if (description) parts.push(description)
  parts.push('You can turn the page when you are ready.')
  return parts.join(' ')
}

function buildDescribeSpeech(result) {
  if (!result) return ''
  if (!result.has_image) {
    return (
      result.image_description ||
      'No important visual image is detected on this page.'
    )
  }
  return result.image_description
    ? result.image_description
    : 'I could not describe this image clearly.'
}

function App() {
  const fileInputRef = useRef(null)
  const videoRef = useRef(null)
  const streamRef = useRef(null)

  const [mode, setMode] = useState('page')
  const [context, setContext] = useState('novel_cover')
  const [cameraOn, setCameraOn] = useState(false)
  const [previewUrl, setPreviewUrl] = useState('')
  const [captureFile, setCaptureFile] = useState(null)
  const [busy, setBusy] = useState(false)
  const [speaking, setSpeaking] = useState(false)
  const [error, setError] = useState('')
  const [assistantLine, setAssistantLine] = useState(
    'Wearable ready. Point at reading material, then tap Capture & Read.',
  )
  const [analyzeResult, setAnalyzeResult] = useState(null)
  const [describeResult, setDescribeResult] = useState(null)

  const activeMode = useMemo(
    () => MODES.find((item) => item.id === mode) || MODES[0],
    [mode],
  )

  useEffect(() => {
    return () => {
      stopSpeaking()
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop())
      }
      if (previewUrl) URL.revokeObjectURL(previewUrl)
    }
  }, [previewUrl])

  async function startCamera() {
    setError('')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: 'environment' } },
        audio: false,
      })
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
      }
      setCameraOn(true)
      setAssistantLine('Camera on. Align the page inside the frame, then capture.')
    } catch {
      setError('Camera not available. You can still upload a photo from the gallery.')
      setAssistantLine('Camera blocked. Use gallery capture instead.')
    }
  }

  function stopCamera() {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop())
      streamRef.current = null
    }
    if (videoRef.current) videoRef.current.srcObject = null
    setCameraOn(false)
  }

  function setCapturedFile(file) {
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    setCaptureFile(file)
    setPreviewUrl(file ? URL.createObjectURL(file) : '')
    setAnalyzeResult(null)
    setDescribeResult(null)
    setError('')
    if (file) {
      setAssistantLine('Page captured. Tap Read Aloud to process with the wearable.')
    }
  }

  function captureFromCamera() {
    const video = videoRef.current
    if (!video) return
    const canvas = document.createElement('canvas')
    canvas.width = video.videoWidth || 1280
    canvas.height = video.videoHeight || 720
    const ctx = canvas.getContext('2d')
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
    canvas.toBlob(
      (blob) => {
        if (!blob) {
          setError('Could not capture from camera.')
          return
        }
        const file = new File([blob], `wearable_capture_${Date.now()}.jpg`, {
          type: 'image/jpeg',
        })
        setCapturedFile(file)
        stopCamera()
      },
      'image/jpeg',
      0.92,
    )
  }

  function onGalleryPick(event) {
    const file = event.target.files?.[0]
    if (!file) return
    stopCamera()
    setCapturedFile(file)
  }

  async function runWearable() {
    if (!captureFile) {
      setError('Capture or upload a page first.')
      setAssistantLine('No page captured yet. Use the camera or gallery.')
      return
    }

    stopSpeaking()
    setBusy(true)
    setSpeaking(false)
    setError('')
    setAnalyzeResult(null)
    setDescribeResult(null)
    setAssistantLine(
      mode === 'page'
        ? 'Reading the page… detecting document type, title, and image.'
        : 'Looking at the selected image… preparing a short description.',
    )

    const formData = new FormData()
    formData.append('file', captureFile)

    try {
      if (mode === 'page') {
        const response = await axios.post(`${API_BASE}/abhishek/analyze`, formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
          timeout: 180000,
        })
        const data = response.data
        setAnalyzeResult(data)
        const speech = buildAnalyzeSpeech(data)
        setAssistantLine(speech)
        speakText(speech, {
          onStart: () => setSpeaking(true),
          onEnd: () => setSpeaking(false),
        })
      } else {
        formData.append('context', context)
        const response = await axios.post(
          `${API_BASE}/abhishek/describe-image`,
          formData,
          {
            headers: { 'Content-Type': 'multipart/form-data' },
            timeout: 180000,
          },
        )
        const data = response.data
        setDescribeResult(data)
        const speech = buildDescribeSpeech(data)
        setAssistantLine(speech)
        speakText(speech, {
          onStart: () => setSpeaking(true),
          onEnd: () => setSpeaking(false),
        })
      }
    } catch (err) {
      const message = backendErrorMessage(err)
      setError(message)
      setAssistantLine(message)
    } finally {
      setBusy(false)
    }
  }

  function replayVoice() {
    const speech =
      mode === 'page'
        ? buildAnalyzeSpeech(analyzeResult)
        : buildDescribeSpeech(describeResult)
    if (!speech) return
    setAssistantLine(speech)
    speakText(speech, {
      onStart: () => setSpeaking(true),
      onEnd: () => setSpeaking(false),
    })
  }

  return (
    <div className="device-app">
      <div className="ambient" aria-hidden="true" />

      <div className="device-frame">
        <div className="device-bezel">
          <div className="device-notch">
            <span className={`pulse-dot ${busy || speaking ? 'active' : ''}`} />
            <span>SWRA Wearable</span>
          </div>

          <div className="device-screen">
            <header className="device-top">
              <div>
                <p className="brand">Smart Wearable Reading Assistant</p>
                <h1>Abhishek Module</h1>
              </div>
              <div className={`live-badge ${speaking ? 'speaking' : busy ? 'busy' : 'idle'}`}>
                {speaking ? 'Speaking' : busy ? 'Reading' : 'Ready'}
              </div>
            </header>

            <p className="device-subtitle">
              Works like the wearable: capture a page, hear the document type,
              title, and image description.
            </p>

            <div className="mode-tabs" role="tablist">
              {MODES.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  role="tab"
                  className={`mode-tab ${mode === item.id ? 'active' : ''}`}
                  onClick={() => {
                    setMode(item.id)
                    setError('')
                    setAssistantLine(item.hint)
                  }}
                >
                  {item.title}
                </button>
              ))}
            </div>

            <p className="mode-hint">{activeMode.hint}</p>

            {mode === 'image' ? (
              <label className="context-row">
                <span>What are you pointing at?</span>
                <select
                  value={context}
                  onChange={(event) => setContext(event.target.value)}
                >
                  {CONTEXT_BY_HINT.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>
            ) : null}

            <div className="viewfinder">
              {cameraOn ? (
                <video ref={videoRef} className="camera-feed" playsInline muted />
              ) : previewUrl ? (
                <img src={previewUrl} alt="Captured page" className="capture-preview" />
              ) : (
                <div className="viewfinder-empty">
                  <div className="scan-frame" />
                  <p>Align reading material in the frame</p>
                </div>
              )}
              <div className="viewfinder-corners" aria-hidden="true" />
            </div>

            <div className="capture-actions">
              {!cameraOn ? (
                <button type="button" className="btn ghost" onClick={startCamera}>
                  Open Camera
                </button>
              ) : (
                <>
                  <button type="button" className="btn ghost" onClick={stopCamera}>
                    Close Camera
                  </button>
                  <button type="button" className="btn primary" onClick={captureFromCamera}>
                    Capture Page
                  </button>
                </>
              )}
              <button
                type="button"
                className="btn ghost"
                onClick={() => fileInputRef.current?.click()}
              >
                Gallery
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                hidden
                onChange={onGalleryPick}
              />
            </div>

            <button
              type="button"
              className="btn power"
              onClick={runWearable}
              disabled={busy}
            >
              {busy
                ? 'Processing…'
                : mode === 'page'
                  ? 'Capture & Read Aloud'
                  : 'Describe & Speak'}
            </button>

            <section className={`assistant-card ${speaking ? 'is-speaking' : ''}`}>
              <div className="assistant-label">Wearable voice</div>
              <p>{assistantLine}</p>
              {(analyzeResult || describeResult) && !busy ? (
                <button type="button" className="btn ghost compact" onClick={replayVoice}>
                  Speak again
                </button>
              ) : null}
            </section>

            {error ? <div className="error-banner">{error}</div> : null}

            {analyzeResult && mode === 'page' ? (
              <section className="result-strip">
                <div>
                  <span>Type</span>
                  <strong>{analyzeResult.document_type || '—'}</strong>
                </div>
                <div>
                  <span>Confidence</span>
                  <strong>{formatConfidence(analyzeResult.confidence) || '—'}</strong>
                </div>
                <div className="wide">
                  <span>Title</span>
                  <strong>{analyzeResult.title || '—'}</strong>
                </div>
                <div className="wide">
                  <span>Image</span>
                  <strong>
                    {joinDescriptions(analyzeResult.image_descriptions) || '—'}
                  </strong>
                </div>
                <div>
                  <span>Status</span>
                  <strong>{analyzeResult.status || '—'}</strong>
                </div>
              </section>
            ) : null}

            {describeResult && mode === 'image' ? (
              <section className="result-strip">
                <div>
                  <span>Has image</span>
                  <strong>{describeResult.has_image ? 'Yes' : 'No'}</strong>
                </div>
                <div>
                  <span>Context</span>
                  <strong>{describeResult.context || '—'}</strong>
                </div>
                <div className="wide">
                  <span>Description</span>
                  <strong>{describeResult.image_description || '—'}</strong>
                </div>
                <div>
                  <span>Status</span>
                  <strong>{describeResult.status || '—'}</strong>
                </div>
              </section>
            ) : null}
          </div>

          <div className="device-home">
            <span />
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
