// Week 4: handle tab audio capture, relay to Flask backend
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === "START_CAPTURE") {
    // TODO: chrome.tabCapture.capture(...)
  }
});
