const app = document.querySelector("#app");
const back = document.querySelector("#back");
const state = { screen: "home", photos: [], stream: null, job: null, unit: "mm", picking: false, picks: [] };

function el(html) {
  const wrap = document.createElement("div");
  wrap.innerHTML = html.trim();
  return wrap.firstElementChild;
}
function show(node) { app.replaceChildren(node); }
function mm(value) {
  if (value == null || Number.isNaN(value)) return "—";
  if (state.unit === "in") return (value / 25.4).toFixed(3) + " in";
  return value.toFixed(2) + " mm";
}

back.onclick = () => {
  stopCamera();
  go("home");
};

function go(screen) {
  state.screen = screen;
  back.hidden = screen === "home";
  const screens = { home, print: printScreen, flat: flatScreen, review: reviewScreen, object: objectScreen, model: modelScreen, history: historyScreen };
  (screens[screen] || home)();
}

async function api(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let detail = response.statusText;
    try { detail = (await response.json()).detail || detail; } catch (_) { /* body wasn't JSON */ }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  const type = response.headers.get("content-type") || "";
  if (type.includes("application/json")) return response.json();
  return response;
}

function home() {
  show(el(`
    <section>
      <h1>Photo to CAD</h1>
      <p class="lead">Turn a phone photo of a real part into geometry SOLIDWORKS can open, at true size.</p>
      <button class="choice" id="go-flat" type="button">
        <strong>Flat part</strong>
        <span>Plates, brackets, gaskets, templates. One or more photos on marker sheets. Exports a 1:1 DXF.</span>
        <span class="tag">Target: under 1 mm on a single sheet</span>
      </button>
      <button class="choice" id="go-object" type="button">
        <strong>3D part</strong>
        <span>Walk around the piece with markers in view. Exports STL and OBJ. Check critical sizes with calipers — phone 3D is about ±1.5 mm on a small part, and the error grows with size.</span>
        <span class="tag">STL + OBJ</span>
      </button>
      <button class="secondary" id="go-print" type="button">Print marker sheets</button>
      <button class="ghost" id="go-history" type="button" style="margin-top:8px">Past jobs</button>
    </section>
  `));
  document.querySelector("#go-flat").onclick = () => { state.photos = []; go("flat"); };
  document.querySelector("#go-object").onclick = () => { state.photos = []; go("object"); };
  document.querySelector("#go-print").onclick = () => go("print");
  document.querySelector("#go-history").onclick = () => go("history");
}

async function printScreen() {
  show(el(`<section><h1>Marker sheets</h1><p class="lead">Print at 100% scale. Do not use Fit to Page. Check the black bar with a ruler before you measure a part.</p><div id="layouts"></div></section>`));
  const data = await api("/api/sheets");
  const host = document.querySelector("#layouts");
  for (const layout of data.layouts) {
    const card = el(`
      <form class="card">
        <h2>${layout.label}</h2>
        <p>${layout.description}</p>
        <p class="fine">${layout.marker_count} markers · ${layout.marker_size_mm} mm squares · scale bar ${layout.scale_bar_mm} mm · up to ${layout.max_sheets} sheets</p>
        <label class="field">Number of sheets
          <span>Each sheet gets its own marker IDs. Use several around a part that does not fit on one page.</span>
          <input name="count" type="number" min="1" max="${layout.max_sheets}" value="1">
        </label>
        <label class="field">First sheet number
          <input name="start" type="number" min="1" max="${layout.max_sheets}" value="1">
        </label>
        <button class="primary" type="submit">Download PDF</button>
      </form>
    `);
    card.onsubmit = (event) => {
      event.preventDefault();
      const count = card.querySelector("[name=count]").value;
      const start = card.querySelector("[name=start]").value;
      window.location = `/api/sheets/pdf?layout=${layout.key}&count=${count}&start=${start}`;
    };
    host.append(card);
  }
}

function photoSection(minCount, tips) {
  return `
    <div class="camera" id="camera" hidden>
      <video id="video" playsinline autoplay muted></video>
    </div>
    <div class="row">
      <button class="primary" id="open-cam" type="button">Open camera</button>
      <button class="secondary" id="shutter" type="button" disabled>Take photo</button>
    </div>
    <label class="secondary" style="display:block;text-align:center;margin-top:8px;padding:14px;border-radius:12px;border:1px solid var(--line);background:#fff">
      Or choose photos
      <input id="files" type="file" accept="image/*" capture="environment" multiple hidden>
    </label>
    <p class="fine" id="count">${state.photos.length} photo${state.photos.length === 1 ? "" : "s"} · ${tips}</p>
    <div class="thumbs" id="thumbs"></div>
    <button class="primary" id="run" type="button" ${state.photos.length >= minCount ? "" : "disabled"} style="margin-top:12px">Measure</button>
  `;
}

function renderThumbs() {
  const box = document.querySelector("#thumbs");
  if (!box) return;
  box.replaceChildren();
  state.photos.forEach((blob, index) => {
    const fig = document.createElement("figure");
    const img = document.createElement("img");
    img.src = URL.createObjectURL(blob);
    const del = document.createElement("button");
    del.type = "button";
    del.textContent = "×";
    del.onclick = () => {
      state.photos.splice(index, 1);
      if (state.screen === "object") objectScreen();
      else flatScreen();
    };
    fig.append(img, del);
    box.append(fig);
  });
}

async function openCamera() {
  const video = document.querySelector("#video");
  document.querySelector("#camera").hidden = false;
  state.stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: { facingMode: { ideal: "environment" }, width: { ideal: 1920 }, height: { ideal: 1080 } },
  });
  video.srcObject = state.stream;
  await video.play();
  document.querySelector("#shutter").disabled = false;
  document.querySelector("#open-cam").textContent = "Camera on";
}

function stopCamera() {
  if (state.stream) {
    state.stream.getTracks().forEach((track) => track.stop());
    state.stream = null;
  }
}

function snap() {
  const video = document.querySelector("#video");
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);
  canvas.toBlob((blob) => {
    state.photos.push(blob);
    if (state.screen === "flat") flatScreen();
    else objectScreen();
  }, "image/jpeg", 0.92);
}

function flatScreen() {
  stopCamera();
  show(el(`
    <section>
      <h1>Flat part</h1>
      <ol class="steps">
        <li>Print a marker sheet at 100% and check the scale bar with a ruler.</li>
        <li>Lay the part flat on the sheet. Keep markers visible around it.</li>
        <li>For a large part, place a numbered set of sheets around it and take overlapping photos. Each photo should show markers from two sheets when you can.</li>
        <li>Hold the phone steady, tap to focus, and avoid glare.</li>
      </ol>
      ${photoSection(1, "one is enough on a single sheet")}
      <button class="secondary" id="sample" type="button">Try the sample plate</button>
      <p id="err"></p>
    </section>
  `));
  wireCapture(1, "flat");
  document.querySelector("#sample").onclick = async () => {
    document.querySelector("#err").innerHTML = `<div class="banner info">Measuring the sample plate…</div>`;
    const job = await api("/api/demo/flat", { method: "POST" });
    await watch(job.id, "review");
  };
}

function objectScreen() {
  stopCamera();
  show(el(`
    <section>
      <h1>3D part</h1>
      <ol class="steps">
        <li>Put marker sheets around the object so several stay in view. On a large piece, spread them out — don't stack them in one corner.</li>
        <li>Take at least 10 photos, 20 or more is safer. Overlap each shot about two thirds with the one before it.</li>
        <li>Walk all the way around, then do a second loop a bit higher.</li>
        <li>Wood grain photographs well. Smooth painted or shiny faces need pencil marks or tape.</li>
      </ol>
      <div class="banner warn">Phone 3D is about ±1.5 mm on a small object and worse as it gets larger. Verify every critical dimension before you cut.</div>
      ${photoSection(8, "10 minimum, 20+ is safer")}
      <button class="secondary" id="sample" type="button">Try the sample block</button>
      <p id="err"></p>
    </section>
  `));
  wireCapture(8, "object");
  document.querySelector("#sample").onclick = async () => {
    document.querySelector("#err").innerHTML = `<div class="banner info">Rebuilding the sample block. This takes a few seconds.</div>`;
    const job = await api("/api/demo/object", { method: "POST" });
    await watch(job.id, "model");
  };
}

function wireCapture(minCount, mode) {
  renderThumbs();
  const err = document.querySelector("#err");
  document.querySelector("#open-cam").onclick = async () => {
    try { await openCamera(); }
    catch (e) { err.innerHTML = `<div class="banner bad">Camera isn't available. Use “choose photos” instead. ${e.message}</div>`; }
  };
  document.querySelector("#shutter").onclick = snap;
  document.querySelector("#files").onchange = (event) => {
    for (const file of event.target.files) state.photos.push(file);
    if (mode === "flat") flatScreen(); else objectScreen();
  };
  document.querySelector("#run").onclick = async () => {
    if (state.photos.length < minCount) return;
    const body = new FormData();
    state.photos.forEach((file, i) => body.append("files", file, `photo-${i}.jpg`));
    err.innerHTML = `<div class="banner info">Measuring…</div>`;
    try {
      stopCamera();
      const job = await api(mode === "flat" ? "/api/jobs/flat" : "/api/jobs/object", { method: "POST", body });
      await watch(job.id, mode === "flat" ? "review" : "model");
    } catch (e) {
      err.innerHTML = `<div class="banner bad">${e.message}</div>`;
    }
  };
}

async function watch(id, next) {
  state.job = { id };
  show(el(`<section><h1>Working</h1><p id="msg">Starting</p><div class="bar"><div id="bar"></div></div></section>`));
  for (;;) {
    const job = await api("/api/jobs/" + id);
    state.job = job;
    const msg = document.querySelector("#msg");
    const bar = document.querySelector("#bar");
    if (msg) msg.textContent = job.message || job.status;
    if (bar) bar.style.width = Math.round((job.progress || 0) * 100) + "%";
    if (job.status === "done") { go(next); return; }
    if (job.status === "error") {
      show(el(`<section><h1>Couldn't measure</h1><div class="banner bad">${job.message}</div><button class="primary" id="again" type="button">Back</button></section>`));
      document.querySelector("#again").onclick = () => go("home");
      return;
    }
    await new Promise((r) => setTimeout(r, 700));
  }
}

function reviewScreen() {
  const job = state.job;
  const result = job.result || {};
  const m = result.measurements || {};
  const acc = result.accuracy || {};
  const holes = (m.holes || []).map((hole) => hole.shape === "circle"
    ? `<div class="stat"><b>${mm(hole.diameter_mm)}</b><small>hole diameter</small></div>`
    : `<div class="stat"><b>profile</b><small>hole, not a circle</small></div>`).join("");
  const quality = result.quality || "fair";
  show(el(`
    <section>
      <h1>Check this</h1>
      <p class="lead">Compare these numbers with calipers before you trust the DXF.</p>
      <img alt="Measured outline" src="/api/jobs/${job.id}/preview.jpg?t=${Date.now()}" style="width:100%;border-radius:12px">
      <div class="stats" style="margin-top:12px">
        <div class="stat"><b>${mm(m.length_mm)}</b><small>overall length</small></div>
        <div class="stat"><b>${mm(m.width_mm)}</b><small>overall width</small></div>
        ${holes}
        <div class="stat"><b>±${mm(acc.estimated_error_mm)}</b><small>estimated accuracy</small></div>
      </div>
      <div class="banner ${quality === "good" ? "good" : quality === "poor" ? "bad" : "warn"}">
        Capture is ${quality}. ${result.marker_count || 0} markers, ${result.sheet_count || 1} sheet${result.sheet_count > 1 ? "s" : ""}, marker mismatch ${mm(result.reproj_mm)}.
        The estimate grows with part size (${((acc.size_fraction || 0.0012) * 100).toFixed(2)}% of the diagonal) plus a little for each extra sheet.
      </div>
      ${(result.warnings || []).map((w) => `<div class="banner warn">${w}</div>`).join("")}
      <div class="row">
        <button class="secondary" id="unit" type="button">Show ${state.unit === "mm" ? "inches" : "millimetres"}</button>
        <a class="primary" style="display:flex;align-items:center;justify-content:center;text-decoration:none;border-radius:12px" id="dxf" href="/api/jobs/${job.id}/result.dxf?unit=${state.unit}">Download DXF</a>
      </div>
      <form class="card" id="refs">
        <h2>Caliper check</h2>
        <p class="fine">Enter one or two real measurements of the overall length and width. One value scales the whole outline evenly. If length and width disagree by more than 0.8%, the outline is stretched to both — circles may stop being round.</p>
        <label class="field">Measured length (${state.unit})<input name="length" type="number" step="0.01" min="0" inputmode="decimal"></label>
        <label class="field">Measured width (${state.unit})<input name="width" type="number" step="0.01" min="0" inputmode="decimal"></label>
        <button class="primary" type="submit">Correct the scale</button>
      </form>
      <button class="secondary" id="touch" type="button">Touch up the outline</button>
      <div id="paint"></div>
    </section>
  `));
  document.querySelector("#unit").onclick = () => { state.unit = state.unit === "mm" ? "in" : "mm"; reviewScreen(); };
  document.querySelector("#refs").onsubmit = async (event) => {
    event.preventDefault();
    const form = new FormData(event.target);
    const toMm = (name) => {
      const raw = form.get(name);
      if (!raw) return null;
      const value = Number(raw);
      return state.unit === "in" ? value * 25.4 : value;
    };
    const body = new FormData();
    const length = toMm("length");
    const width = toMm("width");
    if (length) body.append("measured_length_mm", String(length));
    if (width) body.append("measured_width_mm", String(width));
    try {
      state.job = await api(`/api/jobs/${job.id}/flat-update`, { method: "POST", body });
      reviewScreen();
    } catch (e) {
      event.target.insertAdjacentHTML("beforebegin", `<div class="banner bad">${e.message}</div>`);
    }
  };
  document.querySelector("#touch").onclick = () => startPaint(job.id);
}

async function startPaint(jobId) {
  const host = document.querySelector("#paint");
  host.innerHTML = `<div class="card"><p>Paint <b>add</b> to include the part, <b>erase</b> where the sheet was included by mistake. Then update.</p>
    <div class="preview-wrap" id="stage"><img id="base" alt=""><canvas class="paint" id="overlay"></canvas></div>
    <div class="row" style="margin-top:8px">
      <button type="button" class="secondary" id="mode-add">Add</button>
      <button type="button" class="secondary" id="mode-erase">Erase</button>
    </div>
    <label class="field">Brush<input id="brush" type="range" min="4" max="48" value="16"></label>
    <button class="primary" id="send-mask" type="button">Update shape</button></div>`;
  const base = document.querySelector("#base");
  base.src = `/api/jobs/${jobId}/mosaic.jpg?t=${Date.now()}`;
  await new Promise((resolve) => { base.onload = resolve; });
  const maskImg = new Image();
  maskImg.src = `/api/jobs/${jobId}/mask.png?t=${Date.now()}`;
  await new Promise((resolve) => { maskImg.onload = resolve; });
  const full = document.createElement("canvas");
  full.width = maskImg.naturalWidth;
  full.height = maskImg.naturalHeight;
  const fctx = full.getContext("2d");
  fctx.drawImage(maskImg, 0, 0);
  const overlay = document.querySelector("#overlay");
  overlay.width = full.width;
  overlay.height = full.height;
  const octx = overlay.getContext("2d");
  let mode = "add";
  function redraw() {
    octx.clearRect(0, 0, overlay.width, overlay.height);
    octx.drawImage(full, 0, 0);
    octx.globalCompositeOperation = "source-in";
    octx.fillStyle = "rgba(31, 122, 77, 0.45)";
    octx.fillRect(0, 0, overlay.width, overlay.height);
    octx.globalCompositeOperation = "source-over";
  }
  redraw();
  document.querySelector("#mode-add").onclick = () => { mode = "add"; };
  document.querySelector("#mode-erase").onclick = () => { mode = "erase"; };
  let drawing = false;
  function paint(event) {
    const rect = overlay.getBoundingClientRect();
    const x = (event.clientX - rect.left) * (overlay.width / rect.width);
    const y = (event.clientY - rect.top) * (overlay.height / rect.height);
    const radius = Number(document.querySelector("#brush").value) * (overlay.width / rect.width);
    fctx.fillStyle = mode === "add" ? "#fff" : "#000";
    fctx.beginPath();
    fctx.arc(x, y, radius, 0, Math.PI * 2);
    fctx.fill();
    redraw();
  }
  overlay.onpointerdown = (event) => { drawing = true; overlay.setPointerCapture(event.pointerId); paint(event); };
  overlay.onpointermove = (event) => { if (drawing) paint(event); };
  overlay.onpointerup = () => { drawing = false; };
  document.querySelector("#send-mask").onclick = async () => {
    const blob = await new Promise((resolve) => full.toBlob(resolve, "image/png"));
    const body = new FormData();
    body.append("mask", blob, "mask.png");
    state.job = await api(`/api/jobs/${jobId}/flat-update`, { method: "POST", body });
    reviewScreen();
  };
}

function modelScreen() {
  const job = state.job;
  const result = job.result || {};
  const size = (result.extents && result.extents.size_mm) || [0, 0, 0];
  const acc = result.accuracy || {};
  show(el(`
    <section>
      <h1>3D model</h1>
      <div class="banner warn">${acc.note || "Verify critical dimensions with calipers. Phone 3D is about ±1.5 mm on a small part and worse on a large one."}</div>
      <canvas class="viewer" id="view"></canvas>
      <p class="fine">Drag to turn. Pinch to zoom. One finger only rotates.</p>
      <div class="stats">
        <div class="stat"><b>${mm(size[0])}</b><small>extent X</small></div>
        <div class="stat"><b>${mm(size[1])}</b><small>extent Y</small></div>
        <div class="stat"><b>${mm(size[2])}</b><small>extent Z (height if markers stood up)</small></div>
        <div class="stat"><b>±${mm(acc.estimated_error_mm)}</b><small>estimated accuracy</small></div>
      </div>
      <p class="fine">${result.registered_images || 0} photos aligned · scale from ${result.scaled_by || "unknown"} · ${result.marker_edges || 0} marker edges</p>
      ${(result.warnings || []).map((w) => `<div class="banner warn">${w}</div>`).join("")}
      <div class="row">
        <a class="primary" style="display:flex;align-items:center;justify-content:center;text-decoration:none;border-radius:12px" href="/api/jobs/${job.id}/model.stl">Download STL</a>
        <a class="secondary" style="display:flex;align-items:center;justify-content:center;text-decoration:none;border-radius:12px;border:1px solid var(--line)" href="/api/jobs/${job.id}/model.obj">Download OBJ</a>
      </div>
      <form class="card" id="pick-form">
        <h2>Known distance</h2>
        <p class="fine">Tap two spots on the model, then type the real distance between them (a scale bar, or two points you measured). This sets the scale for the whole model.</p>
        <p id="pick-status" class="fine">Tap the first point.</p>
        <label class="field">Distance (${state.unit})<input name="distance" type="number" step="0.01" min="0" inputmode="decimal"></label>
        <button class="primary" type="submit">Set scale from these points</button>
      </form>
      <form class="card" id="scale-form">
        <h2>Check the scale</h2>
        <p class="fine">Enter a caliper or tape measurement. Width and depth use the longer and shorter footprint sides. Height is the vertical extent after the markers are stood up as the up direction.</p>
        <label class="field">Measured width (${state.unit})<input name="width" type="number" step="0.01" min="0" inputmode="decimal"></label>
        <label class="field">Measured depth (${state.unit})<input name="depth" type="number" step="0.01" min="0" inputmode="decimal"></label>
        <label class="field">Measured height (${state.unit})<input name="height" type="number" step="0.01" min="0" inputmode="decimal"></label>
        <button class="primary" type="submit">Apply</button>
      </form>
      <button class="secondary" id="unit" type="button">Show ${state.unit === "mm" ? "inches" : "millimetres"}</button>
    </section>
  `));
  const canvas = document.querySelector("#view");
  const viewer = new TrueShapeViewer(canvas);
  state.picks = [];
  viewer.onPick = (point) => {
    state.picks.push(point);
    if (state.picks.length > 2) state.picks = [point];
    const status = document.querySelector("#pick-status");
    if (!status) return;
    status.textContent = state.picks.length === 1
      ? "First point set. Tap the second."
      : "Both points set. Enter the real distance and apply.";
  };
  fetch(`/api/jobs/${job.id}/model.stl`).then((r) => r.arrayBuffer()).then((buf) => viewer.load(buf)).catch(() => {});
  document.querySelector("#pick-form").onsubmit = async (event) => {
    event.preventDefault();
    if (state.picks.length < 2) {
      event.target.insertAdjacentHTML("beforebegin", `<div class="banner bad">Tap two points on the model first.</div>`);
      return;
    }
    const raw = new FormData(event.target).get("distance");
    if (!raw) return;
    let distance = Number(raw);
    if (state.unit === "in") distance *= 25.4;
    const [a, b] = state.picks;
    const body = new FormData();
    body.append("distance_mm", String(distance));
    ["ax", "ay", "az"].forEach((name, i) => body.append(name, String(a[i])));
    ["bx", "by", "bz"].forEach((name, i) => body.append(name, String(b[i])));
    try {
      state.job = await api(`/api/jobs/${job.id}/object-scale`, { method: "POST", body });
      state.picks = [];
      modelScreen();
    } catch (e) {
      event.target.insertAdjacentHTML("beforebegin", `<div class="banner bad">${e.message}</div>`);
    }
  };
  document.querySelector("#unit").onclick = () => { state.unit = state.unit === "mm" ? "in" : "mm"; modelScreen(); };
  document.querySelector("#scale-form").onsubmit = async (event) => {
    event.preventDefault();
    const form = new FormData(event.target);
    const toMm = (name) => {
      const raw = form.get(name);
      if (!raw) return null;
      const value = Number(raw);
      return state.unit === "in" ? value * 25.4 : value;
    };
    const body = new FormData();
    const width = toMm("width"), depth = toMm("depth"), height = toMm("height");
    if (width) body.append("measured_width_mm", String(width));
    if (depth) body.append("measured_depth_mm", String(depth));
    if (height) body.append("measured_height_mm", String(height));
    try {
      state.job = await api(`/api/jobs/${job.id}/object-scale`, { method: "POST", body });
      modelScreen();
    } catch (e) {
      event.target.insertAdjacentHTML("beforebegin", `<div class="banner bad">${e.message}</div>`);
    }
  };
}

async function historyScreen() {
  show(el(`<section><h1>Past jobs</h1><div id="list" class="card"><p>Loading…</p></div></section>`));
  const data = await api("/api/jobs");
  const list = document.querySelector("#list");
  if (!data.jobs.length) { list.innerHTML = "<p>Nothing measured yet.</p>"; return; }
  list.innerHTML = "";
  for (const job of data.jobs) {
    const row = el(`<div class="job"><div><strong>${job.mode === "flat" ? "Flat" : "3D"}</strong><div class="fine">${job.created_at.replace("T", " ").replace("+00:00", " UTC")} · ${job.status}</div><div class="fine">${job.message || ""}</div></div><div class="row"></div></div>`);
    const actions = row.querySelector(".row");
    if (job.status === "done") {
      const open = el(`<button class="secondary" type="button">Open</button>`);
      open.onclick = async () => { state.job = await api("/api/jobs/" + job.id); go(job.mode === "flat" ? "review" : "model"); };
      actions.append(open);
    }
    const del = el(`<button class="ghost" type="button">Delete</button>`);
    del.onclick = async () => { await api("/api/jobs/" + job.id, { method: "DELETE" }); historyScreen(); };
    actions.append(del);
    list.append(row);
  }
}

go("home");
