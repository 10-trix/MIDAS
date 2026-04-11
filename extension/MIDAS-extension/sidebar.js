// ── MIDAS sidebar.js ─────────────────────────────────────────────
// Handles: Start/Stop, audio capture (tab or mic),
//          STT via backend, word-by-word avatar animation,
//          fallback demo mode when backend is offline

const BACKEND = 'http://localhost:5000';

// ── State ────────────────────────────────────────────────────────
let isRunning    = false;
let mediaStream  = null;   // mic stream
let mediaRecorder = null;
let audioChunks  = [];
let animQueue    = [];     // words waiting to be signed
let animIndex    = 0;
let animTimer    = null;
let recognition  = null;   // Web Speech API (fallback)
let backendOnline = false;

// ── DOM refs ─────────────────────────────────────────────────────
const startBtn    = document.getElementById('startBtn');
const sourceEl    = document.getElementById('source');
const langEl      = document.getElementById('lang');
const transcriptEl= document.getElementById('transcript');
const queueEl     = document.getElementById('wordQueue');
const currentWordEl = document.getElementById('currentWord');
const bannerEl    = document.getElementById('banner');
const canvas      = document.getElementById('avatarCanvas');
const ctx         = canvas.getContext('2d');

// ── Init ─────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  startBtn.addEventListener('click', toggleRunning);
  pingBackend();
  setInterval(pingBackend, 8000);
  drawIdleHand();
});

// ── Backend ping ─────────────────────────────────────────────────
function pingBackend() {
  fetch(`${BACKEND}/ping`, { signal: AbortSignal.timeout(1500) })
    .then(r => { backendOnline = r.ok; setStatus(r.ok ? 'online' : 'offline'); })
    .catch(() => { backendOnline = false; setStatus('offline'); });
}

function setStatus(s) {
  document.getElementById('statusDot').className = 'dot ' + s;
  document.getElementById('statusText').textContent = s;
}

// ── Start / Stop ─────────────────────────────────────────────────
function toggleRunning() {
  isRunning ? stopAll() : startAll();
}

async function startAll() {
  isRunning = true;
  startBtn.textContent = '⏹ Stop';
  startBtn.classList.add('active');
  hideBanner();

  const source = sourceEl.value;

  if (source === 'tab') {
    // Ask background.js to capture tab audio
    chrome.runtime.sendMessage({ type: 'START_TAB_CAPTURE' }, (response) => {
      if (chrome.runtime.lastError || !response || !response.ok) {
        // Tab capture failed or not supported — fall back to demo
        showBanner('Tab audio capture failed. Running demo mode.', 'info');
        startDemoMode();
      } else {
        showBanner('Tab audio captured. Streaming to backend…', 'info');
        startWaveform();
        // background.js will send transcripts back via chrome.runtime.onMessage
      }
    });
  } else {
    // Microphone
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      if (backendOnline) {
        startMicRecording(mediaStream);
      } else {
        startWebSpeechFallback();
      }
      startWaveform();
    } catch(e) {
      showBanner('Microphone access denied.', 'error');
      stopAll();
      return;
    }
  }
}

function stopAll() {
  isRunning = false;
  startBtn.textContent = '▶ Start';
  startBtn.classList.remove('active');
  stopWaveform();
  stopMicRecording();
  stopWebSpeech();
  stopAnimation();

  // Tell background to stop tab capture
  chrome.runtime.sendMessage({ type: 'STOP_TAB_CAPTURE' });
  hideBanner();
}

// ── Mic Recording → Backend ──────────────────────────────────────
function startMicRecording(stream) {
  audioChunks = [];
  mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });

  mediaRecorder.ondataavailable = (e) => {
    if (e.data.size > 0) audioChunks.push(e.data);
  };

  // Send a chunk every 4 seconds
  mediaRecorder.onstop = sendAudioChunk;
  mediaRecorder.start();

  // Restart every 4s to send chunks
  const chunkInterval = setInterval(() => {
    if (!isRunning) { clearInterval(chunkInterval); return; }
    mediaRecorder.stop();
    mediaRecorder.start();
  }, 4000);
}

function stopMicRecording() {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
  if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }
  mediaRecorder = null;
}

async function sendAudioChunk() {
  if (audioChunks.length === 0) return;
  const blob = new Blob(audioChunks, { type: 'audio/webm' });
  audioChunks = [];

  const formData = new FormData();
  formData.append('audio', blob, 'chunk.webm');
  formData.append('lang', langEl.value);

  try {
    const res = await fetch(`${BACKEND}/audio-to-signs`, {
      method: 'POST',
      body: formData,
      signal: AbortSignal.timeout(8000)
    });
    if (!res.ok) throw new Error('Backend error');
    const data = await res.json();
    // Expected: { transcript: "hello how are you", words: ["HELLO","HOW","ARE","YOU"] }
    handleTranscript(data.transcript, data.words);
  } catch(e) {
    // Backend failed mid-session — switch to Web Speech
    showBanner('Backend unreachable. Switched to browser STT.', 'info');
    startWebSpeechFallback();
  }
}

// ── Web Speech API (fallback when backend offline) ───────────────
function startWebSpeechFallback() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) { startDemoMode(); return; }

  recognition = new SR();
  recognition.continuous = true;
  recognition.interimResults = false;
  recognition.lang = 'en-US';

  recognition.onresult = (e) => {
    for (let i = e.resultIndex; i < e.results.length; i++) {
      if (e.results[i].isFinal) {
        const text = e.results[i][0].transcript.trim();
        const words = text.toUpperCase().split(/\s+/);
        handleTranscript(text, words);
      }
    }
  };

  recognition.onerror = () => startDemoMode();
  recognition.onend   = () => { if (isRunning && !backendOnline) recognition.start(); };
  recognition.start();
}

function stopWebSpeech() {
  if (recognition) { recognition.stop(); recognition = null; }
}

// ── Demo mode (when no backend and no mic) ───────────────────────
const DEMO_SENTENCES = [
  ['HELLO', 'HOW', 'ARE', 'YOU'],
  ['THANK', 'YOU', 'VERY', 'MUCH'],
  ['MY', 'NAME', 'IS', 'MIDAS'],
  ['GOOD', 'MORNING', 'EVERYONE'],
  ['PLEASE', 'HELP', 'ME'],
];
let demoTimer = null;

function startDemoMode() {
  showBanner('Demo mode — no backend needed.', 'info');
  let i = 0;
  const next = () => {
    if (!isRunning) return;
    const words = DEMO_SENTENCES[i % DEMO_SENTENCES.length];
    handleTranscript(words.join(' ').toLowerCase(), words);
    i++;
    demoTimer = setTimeout(next, words.length * 900 + 1200);
  };
  next();
}

// ── Handle incoming text → animate ──────────────────────────────
function handleTranscript(text, words) {
  // Update transcript box
  transcriptEl.textContent = text;
  transcriptEl.className = 'has-text';

  // Build word chips
  animQueue = words;
  animIndex = 0;
  renderWordChips();
  stopAnimation();
  playNextWord();
}

function renderWordChips() {
  queueEl.innerHTML = '';
  animQueue.forEach((w, i) => {
    const chip = document.createElement('div');
    chip.className = 'word-chip pending';
    chip.textContent = w;
    chip.id = `chip-${i}`;
    queueEl.appendChild(chip);
  });
}

function playNextWord() {
  if (animIndex >= animQueue.length) {
    currentWordEl.textContent = '—';
    drawIdleHand();
    return;
  }

  const word = animQueue[animIndex];

  // Update chips
  if (animIndex > 0) {
    const prev = document.getElementById(`chip-${animIndex - 1}`);
    if (prev) prev.className = 'word-chip done';
  }
  const curr = document.getElementById(`chip-${animIndex}`);
  if (curr) curr.className = 'word-chip current';

  currentWordEl.textContent = word;
  drawSignForWord(word);

  animIndex++;
  animTimer = setTimeout(playNextWord, 900);
}

function stopAnimation() {
  if (animTimer) { clearTimeout(animTimer); animTimer = null; }
  if (demoTimer) { clearTimeout(demoTimer); demoTimer = null; }
}

// ── Canvas: Draw hand signs ──────────────────────────────────────
// 21 MediaPipe landmark indices mapped to simplified ASL shapes
// Each shape is defined as finger states: [thumb, index, middle, ring, pinky]
// 0 = closed/down, 1 = extended up, 0.5 = half bent

const SIGN_SHAPES = {
  // Vowels & common words
  'A':       [0.4, 0,   0,   0,   0  ],
  'B':       [0,   1,   1,   1,   1  ],
  'C':       [0.6, 0.6, 0.6, 0.6, 0.6],
  'D':       [0,   1,   0.4, 0.4, 0  ],
  'E':       [0,   0.3, 0.3, 0.3, 0.3],
  'F':       [0.5, 0,   1,   1,   1  ],
  'G':       [0.5, 1,   0,   0,   0  ],
  'H':       [0,   1,   1,   0,   0  ],
  'I':       [0,   0,   0,   0,   1  ],
  'J':       [0,   0,   0,   0,   1  ],
  'K':       [0.5, 1,   1,   0,   0  ],
  'L':       [1,   1,   0,   0,   0  ],
  'M':       [0,   0.2, 0.2, 0.2, 0  ],
  'N':       [0,   0.2, 0.2, 0,   0  ],
  'O':       [0.5, 0.5, 0.5, 0.5, 0.5],
  'P':       [0.5, 1,   1,   0,   0  ],
  'Q':       [0.5, 1,   0,   0,   0  ],
  'R':       [0,   1,   1,   0,   0  ],
  'S':       [0.3, 0,   0,   0,   0  ],
  'T':       [0.3, 0.3, 0,   0,   0  ],
  'U':       [0,   1,   1,   0,   0  ],
  'V':       [0,   1,   1,   0,   0  ],
  'W':       [0,   1,   1,   1,   0  ],
  'X':       [0,   0.5, 0,   0,   0  ],
  'Y':       [1,   0,   0,   0,   1  ],
  'Z':       [0,   1,   0,   0,   0  ],
  // Common words → map to nearby letter shapes
  'HELLO':   [1,   1,   1,   1,   1  ],  // open palm / wave
  'YES':     [0,   0,   0,   0,   0  ],  // fist nod
  'NO':      [0,   1,   1,   0,   0  ],  // index + middle shake
  'PLEASE':  [0.5, 0.5, 0.5, 0.5, 0.5], // flat hand rub
  'THANK':   [0.5, 1,   1,   1,   1  ],  // hand from chin
  'YOU':     [0,   1,   0,   0,   0  ],  // index point
  'MY':      [0,   0,   0,   0,   0  ],  // flat fist to chest
  'NAME':    [0,   1,   1,   0,   0  ],  // U handshape
  'IS':      [0,   0,   0,   0,   1  ],  // I shape
  'GOOD':    [0.5, 1,   1,   1,   1  ],
  'MORNING': [0,   1,   1,   1,   0  ],
  'HELP':    [1,   0,   0,   0,   0  ],  // thumbs up
  'MUCH':    [0.6, 0.6, 0.6, 0.6, 0.6],
  'VERY':    [1,   1,   0,   0,   0  ],
  'HOW':     [0.5, 0.5, 0.5, 0.5, 0.5],
  'ARE':     [0.4, 1,   0,   0,   0  ],
  'EVERYONE':[1,   1,   1,   1,   1  ],
  'ME':      [0,   1,   0,   0,   0  ],
  'MIDAS':   [0.5, 1,   1,   1,   0  ],
};

function getShape(word) {
  if (SIGN_SHAPES[word]) return SIGN_SHAPES[word];
  // Fall back to first letter shape
  const letter = word[0];
  return SIGN_SHAPES[letter] || [0.5, 0.5, 0.5, 0.5, 0.5];
}

function drawSignForWord(word) {
  const shape = getShape(word);
  drawHand(shape, `#${word.charCodeAt(0).toString(16).padStart(2,'0')}b6d4`);
}

function drawIdleHand() {
  drawHand([0.8, 0.8, 0.8, 0.8, 0.8], '#7c3aed');
}

function drawHand(fingerStates, accentColor) {
  const W = canvas.width  = canvas.offsetWidth;
  const H = canvas.height = 200;

  ctx.clearRect(0, 0, W, H);

  // Background gradient
  const bg = ctx.createRadialGradient(W/2, H/2, 10, W/2, H/2, H);
  bg.addColorStop(0, '#13131c');
  bg.addColorStop(1, '#0b0b10');
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, W, H);

  const cx = W / 2;
  const cy = H - 30;
  const palmR = 30;

  // Palm
  ctx.beginPath();
  ctx.ellipse(cx, cy - 10, palmR, palmR * 0.85, 0, 0, Math.PI * 2);
  ctx.fillStyle = 'rgba(124,58,237,0.08)';
  ctx.fill();
  ctx.strokeStyle = accentColor || '#7c3aed';
  ctx.lineWidth = 2;
  ctx.stroke();

  // Finger base positions (angle from center, spread)
  const fingers = [
    { name: 'thumb',  baseX: cx - palmR + 5, baseY: cy - 15, angle: -100, len: 38 },
    { name: 'index',  baseX: cx - palmR * 0.5, baseY: cy - palmR + 4, angle: -80, len: 50 },
    { name: 'middle', baseX: cx,              baseY: cy - palmR + 2, angle: -90, len: 54 },
    { name: 'ring',   baseX: cx + palmR * 0.5, baseY: cy - palmR + 4, angle: -100, len: 50 },
    { name: 'pinky',  baseX: cx + palmR - 5,  baseY: cy - 15,        angle: -110, len: 38 },
  ];

  fingers.forEach((f, i) => {
    const ext = fingerStates[i]; // 0 = closed, 1 = fully extended
    drawFinger(f.baseX, f.baseY, f.angle, f.len, ext, accentColor);
  });

  // Wrist line
  ctx.beginPath();
  ctx.moveTo(cx - palmR, cy + 14);
  ctx.lineTo(cx + palmR, cy + 14);
  ctx.strokeStyle = 'rgba(124,58,237,0.3)';
  ctx.lineWidth = 2;
  ctx.stroke();
}

function drawFinger(bx, by, angleDeg, length, extension, color) {
  const angleRad = (angleDeg * Math.PI) / 180;
  const segLen = length / 3;

  // extension: 1 = fully straight, 0 = fully curled
  // When curled, finger bends forward (toward viewer)
  const curl = 1 - extension;

  let x = bx, y = by;
  const joints = [[x, y]];

  for (let seg = 0; seg < 3; seg++) {
    const bendAngle = angleRad + curl * (seg + 1) * 0.35;
    x += Math.cos(bendAngle) * segLen;
    y += Math.sin(bendAngle) * segLen;
    joints.push([x, y]);
  }

  // Draw segments
  ctx.lineWidth = 3.5 - 0.5 * joints.length * 0.3;
  ctx.strokeStyle = color || '#7c3aed';
  ctx.lineCap = 'round';

  for (let s = 0; s < joints.length - 1; s++) {
    ctx.beginPath();
    ctx.moveTo(joints[s][0], joints[s][1]);
    ctx.lineTo(joints[s+1][0], joints[s+1][1]);
    ctx.lineWidth = 4 - s * 0.8;
    ctx.stroke();

    // Joint dots
    ctx.beginPath();
    ctx.arc(joints[s+1][0], joints[s+1][1], 2.5, 0, Math.PI * 2);
    ctx.fillStyle = color || '#7c3aed';
    ctx.fill();
  }

  // Fingertip glow
  const tip = joints[joints.length - 1];
  const glow = ctx.createRadialGradient(tip[0], tip[1], 1, tip[0], tip[1], 7);
  glow.addColorStop(0, (color || '#7c3aed') + '55');
  glow.addColorStop(1, 'transparent');
  ctx.fillStyle = glow;
  ctx.beginPath();
  ctx.arc(tip[0], tip[1], 7, 0, Math.PI * 2);
  ctx.fill();
}

// ── Waveform animation ───────────────────────────────────────────
function startWaveform() {
  document.querySelectorAll('.wbar').forEach(b => b.classList.add('live'));
}
function stopWaveform() {
  document.querySelectorAll('.wbar').forEach(b => b.classList.remove('live'));
}

// ── Banner ───────────────────────────────────────────────────────
function showBanner(msg, type) {
  bannerEl.textContent = msg;
  bannerEl.className = `banner ${type}`;
}
function hideBanner() {
  bannerEl.className = 'banner';
}

// ── Listen for transcripts from background.js (tab capture) ──────
chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type === 'TRANSCRIPT') {
    const words = msg.text.toUpperCase().trim().split(/\s+/);
    handleTranscript(msg.text, words);
  }
});
