const state = { overview: null, errors: null, cleaning: null };
const titles = { overview: "Pregled projekta", search: "Pretraga", crud: "CRUD demo", errors: "Greške modela", cleaning: "Čišćenje dataseta", presentation: "Prezentacija" };

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const esc = (value) => String(value ?? "").replace(/[&<>'"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[c]));
const number = (value, digits = 0) => Number(value ?? 0).toLocaleString("sr-RS", { maximumFractionDigits: digits, minimumFractionDigits: digits });
const percent = value => `${number(Number(value ?? 0) * 100, 2)}%`;

async function api(path, options = {}) {
  const response = await fetch(path, { headers: { "Content-Type": "application/json", ...(options.headers || {}) }, ...options });
  let data;
  try { data = await response.json(); } catch { data = { error: `HTTP ${response.status}` }; }
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

function toast(message, isError = false) {
  const element = $("#toast"); element.textContent = message; element.className = `toast show${isError ? " error" : ""}`;
  clearTimeout(toast.timer); toast.timer = setTimeout(() => element.className = "toast", 3600);
}

function setBusy(button, busy, label = "Radim...") {
  if (!button) return; if (busy) { button.dataset.original = button.textContent; button.textContent = label; button.disabled = true; }
  else { button.textContent = button.dataset.original || button.textContent; button.disabled = false; }
}

function metric(label, value, note, accent = false) {
  return `<article class="metric-card"><span class="metric-label">${esc(label)}</span><strong class="${accent ? "accent" : ""}">${esc(value)}</strong><small>${esc(note)}</small></article>`;
}

function resultCard(item) {
  const score = item.score == null ? "" : `<span class="score">${Number(item.score).toFixed(4)}</span>`;
  return `<article class="result-card"><img src="/api/image?id=${encodeURIComponent(item.id)}" alt="${esc(item.label)}"><div class="result-card-body"><div class="result-card-head"><h4>ID ${esc(item.id)} · ${esc(item.label)}</h4>${score}</div><p>${esc(item.image_path || "Nema putanje")}</p></div></article>`;
}

function renderCards(target, items, empty = "Nema rezultata.") {
  target.innerHTML = items?.length ? items.map(resultCard).join("") : `<div class="empty-state">${esc(empty)}</div>`;
}

function activateView(name) {
  $$(".nav-item").forEach(item => item.classList.toggle("active", item.dataset.view === name));
  $$(".view").forEach(view => view.classList.toggle("active", view.id === `view-${name}`));
  $("#page-title").textContent = titles[name];
  if (name === "errors") loadErrors();
  if (name === "cleaning") loadCleaning();
  if (name === "crud") refreshOverview();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function renderOverview(data) {
  state.overview = data;
  const q = data.qdrant || {};
  $("#overview-metrics").innerHTML = [
    metric("STL-10 dataset", number(data.dataset.count), `${data.dataset.classes} klasa × 100 slika`, true),
    metric("CLIP embedding", `${data.embeddings.dimension}D`, data.embeddings.model.split("/").pop()),
    metric("Qdrant pointovi", number(q.count), q.connected ? `${q.distance || "Cosine"} · kolekcija aktivna` : "Qdrant nije dostupan", q.connected),
    metric("Tačnost k-NN", data.error_analysis ? percent(data.error_analysis.accuracy) : "—", data.error_analysis ? `${data.error_analysis.error_count} pronađene greške` : "Analiza nije pokrenuta")
  ].join("");
  const resources = [
    ["Metadata za slike", data.files.metadata, `${data.dataset.count} validnih redova`],
    ["CLIP embedding matrica", data.files.embeddings, data.embeddings.shape ? data.embeddings.shape.join(" × ") : "nedostaje"],
    ["Qdrant kolekcija", q.collection_exists, q.collection || "stl10_clip_images"],
    ["Izveštaj grešaka", data.files.error_report, data.error_analysis ? `${data.error_analysis.error_count} greške` : "nije generisan"],
    ["Izveštaj čišćenja", data.files.cleaning_report, data.cleaning ? `${data.cleaning.candidate_pairs} parova` : "nije generisan"],
    ["Očišćena kopija", data.files.cleaned_dataset, data.cleaned_dataset ? `${data.cleaned_dataset.cleaned_count} slika` : "nije generisana"]
  ];
  $("#resource-list").innerHTML = resources.map(([label, ok, note]) => `<div class="check-item"><div><span class="check-icon ${ok ? "" : "missing"}">${ok ? "✓" : "!"}</span><strong>${esc(label)}</strong></div><span class="muted">${esc(note)}</span></div>`).join("");
  const dot = $("#sidebar-status-dot"); dot.className = `status-dot ${q.connected && q.collection_exists ? "online" : "offline"}`;
  $("#sidebar-status").textContent = q.connected && q.collection_exists ? "Qdrant je povezan" : "Qdrant nije dostupan";
  $("#sidebar-count").textContent = q.connected ? `${number(q.count)} pointova · ${q.distance || "Cosine"}` : (q.error || "Pokreni Docker Desktop");
  renderCards($("#demo-points"), data.demo_points, "Nema privremenih demo pointova.");
}

async function refreshOverview(showToast = false) {
  try { renderOverview(await api("/api/overview")); if (showToast) toast("Podaci su osveženi."); }
  catch (error) { toast(error.message, true); }
}

function formData(form) { return Object.fromEntries(new FormData(form).entries()); }

async function submitSearch(event, type) {
  event.preventDefault(); const button = $("button[type=submit]", event.currentTarget); setBusy(button, true);
  try {
    const data = formData(event.currentTarget); let response, items, title;
    if (type === "get") { response = await api(`/api/point?id=${data.id}&with_vector=${data.with_vector === "on"}`); items = [response]; title = `Point ID ${data.id}${response.vector_dimension ? ` · vektor ${response.vector_dimension}D` : ""}`; }
    if (type === "filter") { response = await api(`/api/filter?label=${encodeURIComponent(data.label)}&limit=${data.limit}`); items = response.results; title = `Filter: label = ${data.label}`; }
    if (type === "similar") { response = await api(`/api/similar?id=${data.id}&top_k=${data.top_k}&label=${encodeURIComponent(data.label || "")}`); items = response.results; title = `Najsličnije za ID ${data.id}${data.label ? ` · filter ${data.label}` : ""}`; }
    $("#search-result-title").textContent = title; $("#search-result-count").textContent = `${items.length} rezultata`; renderCards($("#search-results"), items);
  } catch (error) { toast(error.message, true); } finally { setBusy(button, false); }
}

async function crudRequest(event, endpoint) {
  event.preventDefault(); const button = $("button[type=submit]", event.currentTarget); setBusy(button, true);
  try {
    const payload = formData(event.currentTarget); const result = await api(endpoint, { method: "POST", body: JSON.stringify(payload) });
    $("#crud-output").textContent = JSON.stringify(result, null, 2); toast("CRUD operacija je uspešna."); await refreshOverview();
  } catch (error) { $("#crud-output").textContent = `GREŠKA: ${error.message}`; toast(error.message, true); } finally { setBusy(button, false); }
}

async function runJob(button, endpoint, output, after) {
  setBusy(button, true, "Obrada..."); output.classList.remove("hidden"); output.textContent = "Komanda je pokrenuta. Sačekaj završetak...";
  try { const result = await api(endpoint, { method: "POST", body: "{}" }); output.textContent = result.output || JSON.stringify(result, null, 2); if (!result.success && result.success !== undefined) throw new Error("Komanda nije završena uspešno. Pogledaj ispis."); toast("Operacija je završena."); if (after) await after(); }
  catch (error) { output.textContent += `\n\nGREŠKA: ${error.message}`; toast(error.message, true); }
  finally { setBusy(button, false); }
}

async function loadErrors() {
  try {
    const data = await api("/api/errors?limit=100"); state.errors = data; const s = data.summary;
    $("#error-metrics").innerHTML = s ? [metric("Analizirano", number(s.total_images), "STL-10 slika"), metric("Tačno", number(s.correct_predictions), "weighted k-NN"), metric("Greške", number(s.error_count), "za detaljan pregled"), metric("Tačnost", percent(s.accuracy), `k = ${s.k}`, true)].join("") : metric("Analiza", "—", "Pokreni analizu grešaka");
    $("#errors-table").innerHTML = data.errors.length ? data.errors.map(row => `<tr><td>${row.id}</td><td>${esc(row.true_label)}</td><td>${esc(row.predicted_label)}</td><td>${Number(row.prediction_confidence).toFixed(3)}</td><td><span class="tag ${row.diagnosis === "class_confusion" ? "warning" : ""}">${esc(row.diagnosis)}</span></td><td><button class="button small secondary error-detail-button" data-id="${row.id}">Detalji</button></td></tr>`).join("") : `<tr><td colspan="6">Nema generisanih rezultata.</td></tr>`;
    $$(".error-detail-button").forEach(button => button.onclick = () => showErrorDetail(button.dataset.id)); renderConfusion(data.confusion_matrix);
  } catch (error) { toast(error.message, true); }
}

function renderConfusion(matrix) {
  const target = $("#confusion-matrix"); if (!matrix?.rows?.length) { target.innerHTML = `<div class="empty-state">Matrica nije generisana.</div>`; return; }
  const columns = matrix.columns; const labelColumn = columns[0];
  target.innerHTML = `<table class="matrix-table"><thead><tr>${columns.map((column, index) => `<th>${index === 0 ? "Stvarna ↓ / predikcija →" : esc(column)}</th>`).join("")}</tr></thead><tbody>${matrix.rows.map(row => `<tr>${columns.map((column, index) => { const value = row[column]; const cls = index === 0 ? "" : (row[labelColumn] === column ? "matrix-diag" : (Number(value) ? "matrix-error" : "")); return `<${index === 0 ? "th" : "td"} class="${cls}">${esc(value)}</${index === 0 ? "th" : "td"}>`; }).join("")}</tr>`).join("")}</tbody></table>`;
}

async function showErrorDetail(id) {
  try { const data = await api(`/api/error-detail?id=${id}`), error = data.error;
    $("#error-detail").innerHTML = `<img class="detail-image" src="/api/image?id=${id}" alt="ID ${id}"><h3 class="detail-title">ID ${id}: ${esc(error.true_label)} → ${esc(error.predicted_label)}</h3><span class="tag warning">${esc(error.diagnosis)}</span><p class="detail-copy">${esc(error.diagnosis_explanation)}</p><div class="neighbor-list">${data.neighbors.map(n => `<div class="neighbor-item"><img src="/api/image?id=${n.neighbor_id}" alt=""><div><strong>ID ${n.neighbor_id} · ${esc(n.neighbor_label)}</strong><span>rang ${n.rank}</span></div><b class="score">${Number(n.score).toFixed(4)}</b></div>`).join("")}</div>`;
  } catch (error) { toast(error.message, true); }
}

async function loadCleaning() {
  try {
    const data = await api("/api/cleaning"); state.cleaning = data; const s = data.summary;
    $("#cleaning-metrics").innerHTML = s ? [metric("Slike", number(s.total_images), "originalni skup"), metric("Slični parovi", number(s.candidate_pairs), `prag ≥ ${s.candidate_threshold}`), metric("Grupe", number(s.groups), `${s.images_in_groups} slike u grupama`), metric("Strogi kandidati", number(s.recommended_removals), `kopija: ${s.potential_cleaned_count} slika`, true)].join("") : metric("Analiza", "—", "Pokreni čišćenje dataseta");
    $("#groups-table").innerHTML = data.groups.length ? data.groups.map(row => `<tr><td>${row.group_id}</td><td>${row.group_size}</td><td>ID ${row.representative_id}</td><td>${esc(row.labels_in_group)}</td><td>${Number(row.maximum_pair_score).toFixed(4)}</td><td><button class="button small secondary group-detail-button" data-id="${row.group_id}">Prikaži</button></td></tr>`).join("") : `<tr><td colspan="6">Nema generisanih grupa.</td></tr>`;
    $$(".group-detail-button").forEach(button => button.onclick = () => showGroupDetail(button.dataset.id)); renderManifest(data.manifest);
  } catch (error) { toast(error.message, true); }
}

function renderManifest(manifest) {
  $("#cleaning-manifest").innerHTML = manifest ? [
    ["Originalno", manifest.original_count], ["Očišćeno", manifest.cleaned_count], ["Izostavljeno iz kopije", manifest.removed_count], ["ID-evi", (manifest.removed_ids || []).join(", ") || "nema"], ["Originali menjani", manifest.original_files_modified ? "DA" : "NE"]
  ].map(([label, value]) => `<div class="manifest-item"><span>${esc(label)}</span><strong>${esc(value)}</strong></div>`).join("") : `<div class="empty-state">Očišćena kopija još nije generisana.</div>`;
}

async function showGroupDetail(id) {
  try { const data = await api(`/api/cleaning-group?id=${id}`);
    $("#group-detail").innerHTML = `<h3 class="detail-title">Grupa ${data.group_id}</h3><div class="neighbor-list">${data.members.map(m => `<div class="neighbor-item"><img src="/api/image?id=${m.point_id}" alt=""><div><strong>ID ${m.point_id} · ${esc(m.label)}</strong><span>${esc(m.recommended_action)} · prema reprezentantu ${Number(m.similarity_to_representative).toFixed(4)}</span></div><span class="tag ${m.recommended_action === "remove_candidate" ? "error" : m.recommended_action === "keep" ? "success" : "warning"}">${esc(m.recommended_action)}</span></div>`).join("")}</div>`;
  } catch (error) { toast(error.message, true); }
}

async function runQuickDemo() {
  const button = $("#run-quick-demo"), output = $("#quick-demo-output"); setBusy(button, true, "Demo je u toku..."); output.textContent = "POČETAK DEMONSTRACIJE\n\n";
  const log = text => { output.textContent += `${text}\n`; output.scrollTop = output.scrollHeight; };
  try {
    const o = await api("/api/overview"); log(`1. DATASET: ${o.dataset.count} slika, ${o.dataset.classes} klasa.`); log(`   CLIP: ${o.embeddings.dimension}D, normalizacija = ${o.embeddings.normalized}.`); log(`2. QDRANT: ${o.qdrant.count} pointova, ${o.qdrant.distance}, kolekcija ${o.qdrant.collection}.\n`);
    const similar = await api("/api/similar?id=1&top_k=5"); log(`3. SIMILARITY SEARCH za ID 1:`); similar.results.forEach((r,i) => log(`   ${i+1}. ID ${r.id} · ${r.label} · score ${r.score.toFixed(4)}`));
    const filtered = await api("/api/filter?label=dog&limit=5"); log(`\n4. PAYLOAD FILTER label=dog: vraćeno ${filtered.results.length} rezultata.`);
    const errors = await api("/api/errors?limit=30"); log(`5. ANALIZA GREŠAKA: ${errors.summary?.error_count ?? "—"} greške, tačnost ${errors.summary ? percent(errors.summary.accuracy) : "—"}.`);
    if (errors.errors.some(e => Number(e.id) === 3)) { const detail = await api("/api/error-detail?id=3"); log(`   ID 3: ${detail.error.true_label} → ${detail.error.predicted_label}; ${detail.neighbors.length} suseda.`); }
    const cleaning = await api("/api/cleaning"); log(`6. ČIŠĆENJE: ${cleaning.summary?.candidate_pairs ?? "—"} parova, ${cleaning.summary?.groups ?? "—"} grupa, ${cleaning.summary?.recommended_removals ?? "—"} stroga kandidata.`); log(`\nDEMO JE USPEŠNO ZAVRŠEN.`); toast("Brzi demo je završen.");
  } catch (error) { log(`\nGREŠKA: ${error.message}`); toast(error.message, true); } finally { setBusy(button, false); }
}

document.addEventListener("DOMContentLoaded", () => {
  $$(".nav-item").forEach(button => button.onclick = () => activateView(button.dataset.view));
  $("#refresh-all").onclick = () => refreshOverview(true);
  $("#get-form").onsubmit = event => submitSearch(event, "get"); $("#filter-form").onsubmit = event => submitSearch(event, "filter"); $("#similar-form").onsubmit = event => submitSearch(event, "similar");
  $("#create-demo-form").onsubmit = event => crudRequest(event, "/api/crud/create-demo"); $("#update-demo-form").onsubmit = event => crudRequest(event, "/api/crud/update"); $("#delete-demo-form").onsubmit = event => crudRequest(event, "/api/crud/delete");
  $("#refresh-demo").onclick = () => refreshOverview(true); $("#cleanup-demo").onclick = async event => { setBusy(event.currentTarget,true); try { const r = await api("/api/crud/cleanup",{method:"POST",body:"{}"}); $("#crud-output").textContent=JSON.stringify(r,null,2); toast(`Obrisano demo pointova: ${r.deleted_count}`); await refreshOverview(); } catch(e){toast(e.message,true)} finally{setBusy(event.currentTarget,false)} };
  $("#run-tests").onclick = event => runJob(event.currentTarget, "/api/run/tests", $("#tests-output"));
  $("#run-error-analysis").onclick = event => runJob(event.currentTarget, "/api/run/error-analysis", $("#error-job-output"), loadErrors);
  $("#run-cleaning-analysis").onclick = event => runJob(event.currentTarget, "/api/run/cleaning-analysis", $("#cleaning-job-output"), loadCleaning);
  $("#build-cleaned").onclick = event => runJob(event.currentTarget, "/api/run/build-cleaned", $("#cleaning-job-output"), loadCleaning);
  $("#verify-cleaned").onclick = event => runJob(event.currentTarget, "/api/run/verify-cleaned", $("#cleaning-job-output"), loadCleaning);
  $("#run-quick-demo").onclick = runQuickDemo;
  refreshOverview();
});
