(() => {
  const MAX_MB = 25;
  const ALLOWED = [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".webm"];

  const $ = (id) => document.getElementById(id);
  const fileInput = $("fileInput"), dropzone = $("dropzone"), fileCard = $("fileCard");
  const player = $("player"), predictBtn = $("predictBtn"), clientError = $("clientError");
  let currentFile = null, objectUrl = null;

  const fmtTime = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")} (${s.toFixed(1)} s)`;
  const fmtSize = (b) => (b > 1048576 ? (b / 1048576).toFixed(1) + " MB" : Math.max(1, Math.round(b / 1024)) + " KB");
  const pct = (x) => (x * 100).toFixed(1) + "%";

  function showError(msg) { clientError.textContent = msg; clientError.hidden = !msg; }

  function resetResults() { $("result").hidden = true; $("empty").hidden = false; }

  function handleFile(file) {
    showError("");
    resetResults();
    if (!file) return;
    const ext = "." + (file.name.split(".").pop() || "").toLowerCase();
    if (!ALLOWED.includes(ext)) { showError(`Unsupported file type. Use one of: ${ALLOWED.join(", ")}`); return; }
    if (file.size === 0) { showError("This file is empty."); return; }
    if (file.size > MAX_MB * 1048576) { showError(`File is too large (limit ${MAX_MB} MB).`); return; }

    currentFile = file;
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = URL.createObjectURL(file);
    player.src = objectUrl;
    $("fileName").textContent = file.name;
    $("fileSize").textContent = fmtSize(file.size);
    $("fileDuration").textContent = "...";
    fileCard.hidden = false;
    predictBtn.disabled = false;
    drawWaveform(file);
  }

  player.addEventListener("loadedmetadata", () => {
    if (isFinite(player.duration)) $("fileDuration").textContent = fmtTime(player.duration);
  });
  player.addEventListener("error", () => {
    $("fileDuration").textContent = "unknown";
    showError("Your browser cannot play this file, but the server may still be able to read it.");
  });

  // Waveform preview (purely visual; skipped quietly if the browser can't decode the file)
  async function drawWaveform(file) {
    const canvas = $("wave"), ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    try {
      const AC = window.AudioContext || window.webkitAudioContext;
      const ac = new AC();
      const buf = await ac.decodeAudioData(await file.arrayBuffer());
      ac.close();
      const data = buf.getChannelData(0), w = canvas.width, h = canvas.height;
      const step = Math.floor(data.length / (w / 4)) || 1;
      ctx.fillStyle = getComputedStyle(document.body).getPropertyValue("--accent").trim() || "#2b50ff";
      for (let i = 0, x = 0; x < w; i += step, x += 4) {
        let peak = 0;
        for (let j = 0; j < step && i + j < data.length; j++) peak = Math.max(peak, Math.abs(data[i + j]));
        const bar = Math.max(3, peak * h * 0.95);
        ctx.fillRect(x, (h - bar) / 2, 2.5, bar);
      }
    } catch (_) { /* ignore */ }
  }

  // Input handling
  fileInput.addEventListener("change", () => handleFile(fileInput.files[0]));
  dropzone.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); } });
  ["dragenter", "dragover"].forEach((ev) => dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("drag"); }));
  dropzone.addEventListener("drop", (e) => handleFile(e.dataTransfer.files[0]));

  // Predict
  predictBtn.addEventListener("click", async () => {
    if (!currentFile) return;
    showError("");
    predictBtn.disabled = true;
    predictBtn.classList.add("loading");
    predictBtn.querySelector(".btn-label").textContent = "Listening...";
    try {
      const form = new FormData();
      form.append("file", currentFile);
      const res = await fetch("/api/predict", { method: "POST", body: form });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(body.detail || "The server could not process this file.");
      renderResult(body);
    } catch (err) {
      resetResults();
      showError(err.message || "Could not reach the server. Is it running?");
    } finally {
      predictBtn.disabled = false;
      predictBtn.classList.remove("loading");
      predictBtn.querySelector(".btn-label").textContent = "Predict Sound";
    }
  });

  function renderResult(r) {
    $("empty").hidden = true;
    $("result").hidden = false;
    $("predLabel").textContent = r.predicted_label;
    $("predConf").textContent = pct(r.confidence);
    const bar = $("predBar");
    bar.style.width = "0";
    requestAnimationFrame(() => requestAnimationFrame(() => { bar.style.width = (r.confidence * 100) + "%"; }));

    const warn = $("warning");
    warn.hidden = !r.warning;
    warn.textContent = r.warning || "";

    const list = $("topList");
    list.innerHTML = "";
    r.top_predictions.forEach((p) => {
      const li = document.createElement("li");
      const row = document.createElement("div"); row.className = "row";
      const a = document.createElement("span"); a.textContent = p.label;
      const b = document.createElement("span"); b.textContent = pct(p.score);
      row.append(a, b);
      const barWrap = document.createElement("div"); barWrap.className = "bar";
      const fill = document.createElement("i"); fill.style.width = (p.score * 100) + "%";
      barWrap.appendChild(fill);
      li.append(row, barWrap);
      list.appendChild(li);
    });

    const w = r.analyzed_window;
    $("resultNote").textContent =
      `Analysed ${w.length.toFixed(1)} s of audio starting at ${w.start.toFixed(1)} s. ` +
      `Model: ${r.model} (${r.num_classes} sound classes).`;
  }
})();
