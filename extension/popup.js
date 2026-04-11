// ── State ──────────────────────────────────────────────────────
let currentLang = 'ASL';
let currentLang2 = 'ASL';
let cameraActive = false;
let micActive = false;
let videoStream = null;
let currentAudioTab = 'mic';
let sentence = [];
let demoInterval = null;
let confVisible = true;

const DEMO_SIGNS_ASL = ['HELLO', 'YES', 'NO', 'PLEASE', 'THANK YOU', 'SORRY', 'A', 'B', 'C', 'D'];
const DEMO_SIGNS_ISL = ['NAMASTE', 'HAAN', 'NAHI', 'SHUKRIYA', 'MAAF', 'A', 'B', 'C'];

function storageSet(obj) {
  try { if (typeof chrome !== 'undefined' && chrome.storage) chrome.storage.local.set(obj); } catch(e) {}
}
function storageGet(keys, cb) {
  try { if (typeof chrome !== 'undefined' && chrome.storage) { chrome.storage.local.get(keys, cb); return; } } catch(e) {}
  cb({});
}

document.addEventListener('DOMContentLoaded', () => {
  // Mode tabs
  document.getElementById('tabSign2Speech').addEventListener('click', () => switchTab('sign2speech'));
  document.getElementById('tabAudio2Sign').addEventListener('click',  () => switchTab('audio2sign'));
  document.getElementById('tabSettings').addEventListener('click',    () => switchTab('settings'));

  // Panel 1
  document.getElementById('aslBtn').addEventListener('click', () => setLang('ASL'));
  document.getElementById('islBtn').addEventListener('click', () => setLang('ISL'));
  document.getElementById('camBtn').addEventListener('click', toggleCamera);
  document.getElementById('clearBtn').addEventListener('click', clearSentence);
  document.getElementById('speakBtn').addEventListener('click', speakSentence);

  // Panel 2
  document.getElementById('aslBtn2').addEventListener('click', () => setLang2('ASL'));
  document.getElementById('islBtn2').addEventListener('click', () => setLang2('ISL'));
  document.getElementById('audioTabMic').addEventListener('click',  () => setAudioTab('mic'));
  document.getElementById('audioTabText').addEventListener('click', () => setAudioTab('text'));
  document.getElementById('audioTabTab').addEventListener('click',  () => setAudioTab('tab'));
  document.getElementById('micBtn').addEventListener('click', toggleMic);
  document.getElementById('translateBtn').addEventListener('click', translateText);
  document.getElementById('captureTabBtn').addEventListener('click', captureTab);

  // Settings
  document.getElementById('editBackendBtn').addEventListener('click', editBackend);
  document.getElementById('confToggle').addEventListener('change', toggleConf);
  document.getElementById('defASL').addEventListener('click', () => setDefault('ASL'));
  document.getElementById('defISL').addEventListener('click', () => setDefault('ISL'));

  checkBackend();
  setInterval(checkBackend, 8000);

  storageGet(['defaultLang', 'backendUrl'], (data) => {
    if (data.defaultLang) { setLang(data.defaultLang); setLang2(data.defaultLang); }
    if (data.backendUrl) {
      document.getElementById('backendUrlDisplay').textContent =
        data.backendUrl.replace('http://','').replace('https://','');
    }
  });
});

function switchTab(tab) {
  ['sign2speech','audio2sign','settings'].forEach((t, i) => {
    ['tabSign2Speech','tabAudio2Sign','tabSettings'][i].split(',');
  });
  const ids = { 'sign2speech': 'tabSign2Speech', 'audio2sign': 'tabAudio2Sign', 'settings': 'tabSettings' };
  Object.keys(ids).forEach(t => {
    document.getElementById(ids[t]).classList.toggle('active', t === tab);
  });
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.getElementById('panel-' + tab).classList.add('active');
}

function setLang(lang) {
  currentLang = lang;
  document.getElementById('aslBtn').classList.toggle('active', lang === 'ASL');
  document.getElementById('islBtn').classList.toggle('active', lang === 'ISL');
}

function setLang2(lang) {
  currentLang2 = lang;
  document.getElementById('aslBtn2').classList.toggle('active', lang === 'ASL');
  document.getElementById('islBtn2').classList.toggle('active', lang === 'ISL');
}

function setDefault(lang) {
  document.getElementById('defASL').classList.toggle('active', lang === 'ASL');
  document.getElementById('defISL').classList.toggle('active', lang === 'ISL');
  storageSet({ defaultLang: lang });
}

async function toggleCamera() {
  if (!cameraActive) {
    try {
      videoStream = await navigator.mediaDevices.getUserMedia({ video: true });
      const video = document.getElementById('videoFeed');
      video.srcObject = videoStream;
      video.style.display = 'block';
      document.getElementById('camPlaceholder').style.display = 'none';
      document.getElementById('camOverlay').classList.add('visible');
      cameraActive = true;
      const btn = document.getElementById('camBtn');
      btn.innerHTML = '<span class="btn-icon">⏹️</span> Stop Camera';
      btn.classList.add('recording');
      if (confVisible) document.getElementById('confRow').style.display = 'flex';
      startDemoDetection();
    } catch (e) {
      setSignOutput('Camera access denied', true);
    }
  } else {
    stopCamera();
  }
}

function stopCamera() {
  if (videoStream) { videoStream.getTracks().forEach(t => t.stop()); videoStream = null; }
  document.getElementById('videoFeed').style.display = 'none';
  document.getElementById('camPlaceholder').style.display = 'flex';
  document.getElementById('camOverlay').classList.remove('visible');
  document.getElementById('confRow').style.display = 'none';
  cameraActive = false;
  const btn = document.getElementById('camBtn');
  btn.innerHTML = '<span class="btn-icon">📷</span> Start Camera';
  btn.classList.remove('recording');
  stopDemoDetection();
  setSignOutput('Waiting for detection…', true);
}

function startDemoDetection() {
  let tick = 0, pending = null, holdCount = 0;
  const HOLD_FRAMES = 5;
  demoInterval = setInterval(() => {
    tick++;
    const signs = currentLang === 'ASL' ? DEMO_SIGNS_ASL : DEMO_SIGNS_ISL;
    if (tick % 14 === 0) {
      pending = signs[Math.floor(Math.random() * signs.length)];
      holdCount = 0;
      setConfidence(55 + Math.floor(Math.random() * 40));
      setSignOutput(pending + ' ?', false);
    }
    if (pending && tick % 14 > 8) {
      if (++holdCount >= HOLD_FRAMES) { confirmSign(pending); pending = null; }
    }
  }, 150);
}

function stopDemoDetection() {
  if (demoInterval) { clearInterval(demoInterval); demoInterval = null; }
}

function confirmSign(sign) {
  setSignOutput(sign, false);
  setConfidence(75 + Math.floor(Math.random() * 25));
  sentence.push(sign);
  updateSentence();
  if (document.getElementById('autoSpeakToggle').checked) speak(sign);
}

function setSignOutput(text, empty) {
  const el = document.getElementById('signOutput');
  el.textContent = text;
  el.className = 'output-text' + (empty ? ' empty' : '');
}

function updateSentence() {
  const el = document.getElementById('sentenceOutput');
  if (sentence.length === 0) { el.textContent = '—'; el.className = 'output-text empty'; }
  else { el.textContent = sentence.join(' '); el.className = 'output-text'; }
}

function clearSentence() {
  sentence = [];
  updateSentence();
  setSignOutput('Waiting for detection…', true);
  setConfidence(0);
}

function setConfidence(pct) {
  document.getElementById('confFill').style.width = pct + '%';
  document.getElementById('confValue').textContent = pct + '%';
}

function toggleConf() {
  confVisible = document.getElementById('confToggle').checked;
  document.getElementById('confRow').style.display = (cameraActive && confVisible) ? 'flex' : 'none';
}

function speak(text) {
  if ('speechSynthesis' in window) {
    const utt = new SpeechSynthesisUtterance(text);
    utt.rate = 0.9;
    window.speechSynthesis.speak(utt);
  }
}

function speakSentence() {
  if (sentence.length > 0) speak(sentence.join(' '));
}

function setAudioTab(tab) {
  currentAudioTab = tab;
  document.getElementById('audioTabMic').classList.toggle('active',  tab === 'mic');
  document.getElementById('audioTabText').classList.toggle('active', tab === 'text');
  document.getElementById('audioTabTab').classList.toggle('active',  tab === 'tab');
  document.getElementById('micUI').style.display  = tab === 'mic'  ? 'block' : 'none';
  document.getElementById('textUI').style.display = tab === 'text' ? 'block' : 'none';
  document.getElementById('tabUI').style.display  = tab === 'tab'  ? 'block' : 'none';
  if (tab !== 'mic' && micActive) stopMic();
}

let recognition = null;

function toggleMic() { micActive ? stopMic() : startMic(); }

function startMic() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) { setTranscript('Speech recognition not supported.', true); return; }
  recognition = new SR();
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.lang = 'en-US';
  recognition.onresult = (e) => {
    let t = '';
    for (let i = e.resultIndex; i < e.results.length; i++) t += e.results[i][0].transcript;
    setTranscript(t, false);
    demoSignFromText(t);
  };
  recognition.onerror = () => stopMic();
  recognition.onend = () => { if (micActive) recognition.start(); };
  recognition.start();
  micActive = true;
  document.getElementById('micBtn').innerHTML = '<span class="btn-icon">⏹️</span> Stop Listening';
  document.getElementById('micBtn').classList.add('recording');
  document.querySelectorAll('.wave-bar').forEach(b => b.classList.add('active'));
}

function stopMic() {
  if (recognition) { recognition.stop(); recognition = null; }
  micActive = false;
  document.getElementById('micBtn').innerHTML = '<span class="btn-icon">🎙️</span> Start Listening';
  document.getElementById('micBtn').classList.remove('recording');
  document.querySelectorAll('.wave-bar').forEach(b => b.classList.remove('active'));
}

function translateText() {
  const text = document.getElementById('textInputField').value.trim();
  if (!text) return;
  setTranscript(text, false);
  demoSignFromText(text);
}

function captureTab() {
  setTranscript('Tab audio capture requires backend connection.', true);
}

function setTranscript(text, empty) {
  const el = document.getElementById('transcriptOutput');
  el.textContent = text || '—';
  el.className = 'output-text' + (empty ? ' empty' : '');
}

function demoSignFromText(text) {
  const words = text.toUpperCase().trim().split(/\s+/);
  let i = 0;
  const playNext = () => {
    if (i >= words.length) return;
    const word = words[i++];
    document.getElementById('currentSignBadge').textContent = word;
    animateHandForSign(word);
    setTimeout(playNext, 900);
  };
  playNext();
}

function animateHandForSign(sign) {
  const g = document.getElementById('handGroup');
  g.style.transition = 'opacity 0.15s';
  g.style.opacity = '0.3';
  setTimeout(() => {
    g.style.opacity = '0.85';
    const offset = (sign.charCodeAt(0) % 5) - 2;
    g.querySelectorAll('line').forEach((l, idx) => {
      l.setAttribute('y2', parseInt(l.getAttribute('y2')) + offset * (idx % 3));
    });
  }, 150);
}

function editBackend() {
  const url = prompt('Enter backend URL:', 'http://localhost:5000');
  if (url) {
    document.getElementById('backendUrlDisplay').textContent =
      url.replace('http://','').replace('https://','');
    storageSet({ backendUrl: url });
  }
}

function checkBackend() {
  fetch('http://localhost:5000/ping', { signal: AbortSignal.timeout(1500) })
    .then(r => setStatus(r.ok ? 'online' : 'offline'))
    .catch(() => setStatus('offline'));
}

function setStatus(s) {
  document.getElementById('statusDot').className = 'dot ' + s;
  document.getElementById('statusText').textContent = s;
}
