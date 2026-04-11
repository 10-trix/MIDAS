// MIDAS Content Script
// Injected into pages — used for tab audio capture when backend is connected

// Stub: listens for messages from background worker
chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type === 'CAPTURE_TAB_AUDIO') {
    // Will hook into page audio context when backend is ready
    console.log('[MIDAS] Tab audio capture stub triggered');
  }
});
