// Week 4: connects to Flask backend, renders avatar frames on canvas
const btn = document.getElementById("startBtn");
btn.addEventListener("click", () => {
  const source = document.getElementById("source").value;
  const lang   = document.getElementById("lang").value;
  console.log(`[SLT] Starting — source: ${source}, lang: ${lang}`);
  // TODO: send audio to localhost:5000/audio-to-signs
});
