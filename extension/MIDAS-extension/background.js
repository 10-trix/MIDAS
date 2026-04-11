// ── MIDAS background.js ──────────────────────────────────────────
// Handles tab audio capture using chrome.tabCapture API
// Sends audio chunks to Flask backend → relays transcript to sidebar

const BACKEND = 'http://localhost:5000';

let mediaRecorder = null;
let audioChunks   = [];
let captureStream = null;
let chunkInterval = null;
let activeLang    = 'ASL';

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === 'START_TAB_CAPTURE') {
    activeLang = msg.lang || 'ASL';
    startTabCapture(sendResponse);
    return true; // async
  }
  if (msg.type === 'STOP_TAB_CAPTURE') {
    stopTabCapture();
    sendResponse({ ok: true });
  }
});

function startTabCapture(sendResponse) {
  chrome.tabCapture.capture({ audio: true, video: false }, (stream) => {
    if (chrome.runtime.lastError || !stream) {
      console.warn('[MIDAS] tabCapture failed:', chrome.runtime.lastError?.message);
      sendResponse({ ok: false });
      return;
    }

    captureStream = stream;
    audioChunks = [];

    // Play silently so YouTube doesn't pause
    const audio = new Audio();
    audio.srcObject = stream;
    audio.volume = 0;
    audio.play();

    mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
    mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) audioChunks.push(e.data); };
    mediaRecorder.onstop = () => sendChunkToBackend();
    mediaRecorder.start();
    sendResponse({ ok: true });

    // Chunk every 4s
    chunkInterval = setInterval(() => {
      if (!captureStream) return;
      if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
        mediaRecorder.start();
      }
    }, 4000);
  });
}

function stopTabCapture() {
  if (chunkInterval) { clearInterval(chunkInterval); chunkInterval = null; }
  if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
  if (captureStream) { captureStream.getTracks().forEach(t => t.stop()); captureStream = null; }
  mediaRecorder = null;
  audioChunks = [];
}

async function sendChunkToBackend() {
  if (audioChunks.length === 0) return;
  const blob = new Blob(audioChunks, { type: 'audio/webm' });
  audioChunks = [];

  const formData = new FormData();
  formData.append('audio', blob, 'chunk.webm');
  formData.append('lang', activeLang);

  try {
    const res = await fetch(`${BACKEND}/audio-to-signs`, {
      method: 'POST',
      body: formData,
      signal: AbortSignal.timeout(8000)
    });
    if (!res.ok) throw new Error('bad response');
    const data = await res.json();
    if (data.transcript) {
      chrome.runtime.sendMessage({
        type: 'TRANSCRIPT',
        text: data.transcript,
        words: data.words || data.transcript.toUpperCase().split(/\s+/)
      });
    }
  } catch(e) {
    console.warn('[MIDAS] Backend error:', e.message);
  }
}

console.log('[MIDAS] Background worker ready');
