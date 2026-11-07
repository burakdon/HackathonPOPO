// CueAgain web app: patient practice, clinician view, models & results. Plain JS, served by the
// agent service at /app (same origin, so no CORS setup is needed).
"use strict";

const $ = (id) => document.getElementById(id);
const show = (el, on = true) => (typeof el === "string" ? $(el) : el).classList.toggle("hidden", !on);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

async function api(path, opts = {}) {
  const r = await fetch(path, { headers: opts.body instanceof FormData ? {} : { "Content-Type": "application/json" }, ...opts });
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail || r.statusText));
  return body;
}
const post = (path, data) => api(path, { method: "POST", body: JSON.stringify(data ?? {}) });

const ERR_LABEL = { sound: "sound error", meaning: "meaning error", no_response: "no response", unclassified: "incorrect, type unclear" };
const HINT_LABEL = { none: "no hint", first_sound: "first sound", first_letter: "first letter", sentence_frame: "sentence", semantic: "description" };
const actionLabel = (a) => a.operation === "retry_current" ? `retry, ${HINT_LABEL[a.cue]}` : `next picture, tier ${a.tier}`;

// ------------------------------------------------------------------ navigation
let currentView = "practice";
document.querySelectorAll(".nav").forEach((b) => b.addEventListener("click", () => {
  currentView = b.dataset.view;
  document.querySelectorAll(".nav").forEach((x) => x.classList.toggle("active", x === b));
  document.querySelectorAll(".view").forEach((v) => show(v, v.id === `view-${currentView}`));
  if (currentView === "clinician") loadClinician();
  if (currentView === "models") loadModels();
}));

// ------------------------------------------------------------------ practice
const P = { sid: null, pres: null, shownAt: 0, recorder: null, chunks: [], heard: null, attemptId: null, correct: null };

async function initPractice() {
  try { const m = await api("/models"); $("start-model").textContent = m.active; } catch (e) { $("start-model").textContent = "unknown"; }
}

$("btn-start").onclick = async () => {
  try {
    const v = await post("/practice/sessions", {});
    P.sid = v.session_id;
    show("start-card", false); show("practice");
    renderSession(v); renderPresentation(v.presentation, null);
  } catch (e) { alert(`Could not start: ${e.message}`); }
};

function renderSession(v) {
  $("p-session").textContent = v.session_id;
  $("p-model").textContent = v.model_version;
}

function renderPresentation(p, decision) {
  P.pres = p; P.shownAt = performance.now();
  $("p-image").src = p.image_url;
  $("p-image").alt = "Picture to name";
  show("p-hint", !!p.hint_text);
  $("p-hint").textContent = p.hint_text ? `Hint: ${p.hint_text}` : "";
  const msg = $("p-message");
  if (decision) {
    const a = decision.shown_action;
    msg.textContent = a.operation === "retry_current" ? "Let's try this one again." : "Here is a new picture.";
    const d = decision.decision;
    $("p-source").innerHTML = ` · last decision by <b>${esc(d.source)}</b>` +
      (d.fallback_reason ? ` <span class="chip warnc">fallback: ${esc(d.fallback_reason)}</span>` : "") +
      (decision.overridden ? ` <span class="chip">clinician choice</span>` : "");
    show(msg);
  } else { show(msg, false); }
  resetAnswer();
}

function resetAnswer() {
  P.heard = null; P.attemptId = null; P.correct = null;
  show("helper", false); show("h-cats", false); show("btn-record"); show("btn-stop", false);
  show("rec-state", false); show("busy", false); show("mic-error", false);
}

function pickMime() {
  const c = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg;codecs=opus"];
  return c.find((m) => window.MediaRecorder && MediaRecorder.isTypeSupported(m)) || "";
}

$("btn-record").onclick = async () => {
  show("mic-error", false);
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });   // only after the user presses
  } catch (e) {
    $("mic-error").textContent = `Microphone not available (${e.name}). You can type the answer instead; it will be labelled as manual entry.`;
    show("mic-error"); return;
  }
  const mime = pickMime();
  P.chunks = [];
  P.recorder = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
  P.recorder.ondataavailable = (ev) => ev.data.size && P.chunks.push(ev.data);
  P.recorder.onstop = () => { stream.getTracks().forEach((t) => t.stop()); transcribe(); };
  P.recorder.start();
  show("btn-record", false); show("btn-stop"); show("rec-state");
};

$("btn-stop").onclick = () => {
  P.responseTime = (performance.now() - P.shownAt) / 1000;
  show("btn-stop", false); show("rec-state", false); show("busy");
  P.recorder.stop();
};

async function transcribe() {
  const blob = new Blob(P.chunks, { type: P.recorder.mimeType || "audio/webm" });
  const fd = new FormData();
  fd.append("audio", blob, blob.type.includes("mp4") ? "attempt.mp4" : "attempt.webm");
  fd.append("exercise_id", P.pres.exercise_id);
  let res;
  try { res = await api("/transcribe", { method: "POST", body: fd }); }
  catch (e) { res = { status: "failed", text: "", provider: "network", error: e.message }; }
  show("busy", false);
  openHelper(res);
}

function openHelper(res) {
  P.heard = res;
  P.attemptId = crypto.randomUUID();     // one id per answer: a resubmission cannot be counted twice
  show("helper"); show("h-cats", false); show("btn-record", false);
  const ok = res.status === "ok";
  $("h-recog").innerHTML = res.status === "manual"
    ? `<span class="chip warnc">manual entry</span> no recording was used`
    : ok ? `Speech recogniser (${esc(res.provider)}) heard: <b>“${esc(res.text)}”</b>`
         : `<span class="chip warnc">${esc(res.status)}</span> ${esc(res.error || "no transcript")}. This is a recording problem, not a wrong answer.`;
  $("h-text").value = ok ? res.text : "";
  const s = res.suggestion;
  $("h-suggest").innerHTML = s ? (s.suggestion === "matches"
    ? `Suggestion: looks like “${esc(s.matched_name)}”. <span class="muted">${esc(s.note)}</span>`
    : `Suggestion: does not look like “${esc(P.pres.word)}”. <span class="muted">${esc(s.note)}</span>`) : "";
  $("h-text").focus();
}

$("btn-type").onclick = () => { if (!P.pres) return; P.responseTime = (performance.now() - P.shownAt) / 1000; openHelper({ status: "manual", text: "", provider: "manual" }); };
$("h-correct").onclick = () => submitAttempt(true, null);
$("h-incorrect").onclick = () => show("h-cats");
document.querySelectorAll("#h-cats button").forEach((b) => (b.onclick = () => submitAttempt(false, b.dataset.cat)));
$("h-unscored").onclick = () => submitAttempt(null, null, true);

async function submitAttempt(correct, category, unscored = false) {
  const h = P.heard || { status: "manual", text: "", provider: "manual" };
  const body = {
    attempt_id: P.attemptId || crypto.randomUUID(), presentation_id: P.pres.presentation_id,
    raw_transcript: h.text || null, transcript_status: h.status, transcript_provider: h.provider,
    corrected_transcript: $("h-text").value.trim() || null,
    scoring_status: unscored ? "unscored" : "confirmed",
    correct: unscored ? null : correct, error_category: unscored ? null : category,
    unscored_reason: unscored ? (h.status !== "ok" ? `recording ${h.status}` : "helper could not tell") : null,
    response_time_s: Math.min(600, Math.round((P.responseTime || 0) * 10) / 10),
  };
  P.attemptId = body.attempt_id;
  document.querySelectorAll("#helper button").forEach((b) => (b.disabled = true));
  try {
    const r = await post(`/practice/sessions/${P.sid}/attempts`, body);
    if (r.decision) renderPresentation(r.presentation, r.decision);
    else { resetAnswer(); $("p-message").textContent = "Not scored. Let's try the same picture again."; show("p-message"); }
  } catch (e) { alert(`Could not save the answer (you can press the button again; it will not be counted twice): ${e.message}`); }
  finally { document.querySelectorAll("#helper button").forEach((b) => (b.disabled = false)); }
}

$("btn-hint").onclick = async () => {
  try { const v = await post(`/practice/sessions/${P.sid}/manual`, { presentation_id: P.pres.presentation_id, kind: "show_hint", actor: "patient" });
    const keep = P.shownAt; renderPresentation(v.presentation, null); P.shownAt = keep; }
  catch (e) { alert(e.message); }
};
$("btn-skip").onclick = async () => {
  try { const v = await post(`/practice/sessions/${P.sid}/manual`, { presentation_id: P.pres.presentation_id, kind: "skip", actor: "patient" });
    renderPresentation(v.presentation, null); }
  catch (e) { alert(e.message); }
};
$("btn-pause").onclick = async () => { await post(`/practice/sessions/${P.sid}/pause?paused=true`); show("paused"); };
$("btn-resume").onclick = async () => { await post(`/practice/sessions/${P.sid}/pause?paused=false`); show("paused", false); P.shownAt = performance.now(); };

// ------------------------------------------------------------------ clinician
let clinTimer = null;
async function loadClinician() {
  const list = await api("/practice/sessions");
  const sel = $("c-session");
  const prev = sel.value;
  sel.innerHTML = list.map((s) => `<option value="${esc(s.session_id)}">${esc(s.session_id)} · ${esc(s.model_version)} · ${s.confirmed_attempts} answers</option>`).join("");
  if (prev && list.some((s) => s.session_id === prev)) sel.value = prev;
  else if (P.sid) sel.value = P.sid;
  show("c-empty", !list.length); show("c-body", !!list.length);
  if (list.length) await renderClinician();
  clearTimeout(clinTimer);
  clinTimer = setTimeout(() => currentView === "clinician" && !document.activeElement.closest?.("#c-choose") && loadClinician(), 4000);
}
$("c-session").onchange = renderClinician;

let C = { session: null, decision: null };
async function renderClinician() {
  const sid = $("c-session").value;
  const s = await api(`/practice/sessions/${sid}`);
  C.session = s;
  const st = s.stats;
  $("c-indep").textContent = `${st.independent_naming.correct} / ${st.independent_naming.n}`;
  $("c-hinted").textContent = `${st.naming_after_hint.correct} / ${st.naming_after_hint.n}`;
  $("c-fallbacks").textContent = `${st.fallbacks} / ${st.decisions}`;
  const decs = s.decisions;
  const d = decs[decs.length - 1];
  const keepSelection = C.decision && d && C.decision.decision_id === d.decision_id;
  C.decision = d || null;
  if (!d) { $("c-latest").textContent = "No decision yet: the first picture is fixed."; show("c-choose", false); }
  else {
    const att = s.attempts.find((a) => a.attempt_id === d.after_attempt_id);
    const dd = d.decision;
    const res = att.correct ? "correct" : `not correct (${ERR_LABEL[att.error_category]})`;
    const pending = s.presentation.decision_id === d.decision_id && !s.attempts.some((a) => a.presentation_id === s.presentation.presentation_id);
    $("c-latest").innerHTML = `<div>After <b>${esc(att.word)}</b> (${HINT_LABEL[att.hint]}): <b>${esc(res)}</b>, ${att.response_time_s.toFixed(1)} s.</div>
      <div style="font-size:20px;margin:8px 0">Proposal: <b>${esc(d.proposal_label)}</b>
      ${d.overridden ? `<span class="chip">replaced by clinician: ${esc(actionLabel(d.shown_action))}</span>` : ""}</div>
      <div class="small">Decided by <b>${esc(dd.source)}</b>${dd.requested_model !== dd.source ? ` (requested ${esc(dd.requested_model)})` : ""}
      ${dd.fallback_reason ? `<span class="chip warnc">fallback: ${esc(dd.fallback_reason)}</span>` : ""}
      · ${dd.latency_ms.toFixed(0)} ms ${dd.raw_output ? `· model output <code>${esc(dd.raw_output)}</code>` : ""}</div>
      <div class="small muted">${pending ? "The patient has not answered this picture yet: your choice will replace it." : "Already shown and answered: a different choice is saved as an example only."}</div>`;
    show("c-choose");
    if (!keepSelection) {
      $("c-options").innerHTML = d.options.map((o) => `<label class="opt ${sameAction(o, dd.action) ? "proposed" : ""}">
        <input type="radio" name="copt" value="${esc(o.key)}" ${sameAction(o, dd.action) ? "checked" : ""}>
        <span>${esc(o.label)}${sameAction(o, dd.action) ? ' <span class="chip">current proposal</span>' : ""}</span></label>`).join("");
      $("c-feedback").textContent = "";
    }
    $("c-use").textContent = pending ? "Use my choice" : "Save my choice as an example";
  }
  $("c-rows").innerHTML = s.attempts.slice().reverse().slice(0, 15).map((a) => {
    const dec = decs.find((x) => x.after_attempt_id === a.attempt_id);
    const res = a.scoring_status === "unscored" ? `<span class="chip warnc">not scored</span> ${esc(a.unscored_reason || "")}`
      : a.correct ? '<span class="ok">correct</span>' : `<span class="miss">not correct</span> (${ERR_LABEL[a.error_category]})`;
    const heard = `${a.transcript_status === "manual" ? '<span class="chip">typed</span>' : esc(a.raw_transcript || "–")} → ${esc(a.corrected_transcript || "–")}`;
    return `<tr><td>${esc(a.word)}</td><td>${HINT_LABEL[a.hint]}</td><td>${heard}</td><td>${res}</td>
      <td>${dec ? esc(actionLabel(dec.shown_action)) + (dec.overridden ? " (clinician)" : "") : "–"}</td>
      <td class="small">${dec ? esc(dec.decision.source) + (dec.decision.fallback_reason ? " †" : "") : ""}</td></tr>`;
  }).join("");
  const demos = await api("/demonstrations");
  $("c-demos").innerHTML = demos.slice(0, 15).map((r) => {
    const a = JSON.parse(r.preferred);
    const lab = a.op === "retry" ? `retry, ${HINT_LABEL[a.hint]}` : `next picture, tier ${a.tier}`;
    return `<tr><td class="small">${esc(r.created.slice(11, 19))}</td><td>${esc(lab)}${r.provenance.accepted_proposal ? ' <span class="chip">accepted proposal</span>' : ""}</td>
      <td>${esc(r.scope)}</td><td class="small">${esc(r.note || "")}</td>
      <td class="small"><span class="chip ${r.training_status.startsWith("pending") ? "warnc" : ""}">${esc(r.training_status)}</span></td></tr>`;
  }).join("") || '<tr><td colspan="5" class="muted">None yet.</td></tr>';
}
function sameAction(o, a) { return o.operation === a.operation && o.cue === a.cue && (o.tier ?? null) === (a.tier ?? null); }

function chosenOption() {
  const v = document.querySelector('input[name="copt"]:checked');
  return v && C.decision.options.find((o) => o.key === v.value);
}
$("c-use").onclick = async () => {
  const o = chosenOption(); if (!o) return;
  const s = C.session, d = C.decision;
  const action = { operation: o.operation, cue: o.cue, tier: o.tier };
  const pending = s.presentation.decision_id === d.decision_id;
  try {
    if (pending) {
      const r = await post(`/practice/sessions/${s.session_id}/override`, { decision_id: d.decision_id, action, note: $("c-note").value || null, scope: $("c-scope").value });
      $("c-feedback").textContent = `Done: the patient now sees “${o.label}”. Example saved (pending, not in any trained model).`;
      if (P.sid === s.session_id) renderPresentation(r.presentation, { ...d, shown_action: action, overridden: true });
    } else {
      await post("/demonstrations", { session_id: s.session_id, decision_id: d.decision_id, action, note: $("c-note").value || null, scope: $("c-scope").value });
      $("c-feedback").textContent = "Saved as an example (pending, not in any trained model).";
    }
    $("c-note").value = ""; C.decision = null; await renderClinician();
  } catch (e) { $("c-feedback").textContent = `Not saved: ${e.message}`; }
};
$("c-accept").onclick = async () => {
  const s = C.session, d = C.decision; const a = d.decision.action;
  try {
    await post("/demonstrations", { session_id: s.session_id, decision_id: d.decision_id, action: a, accepted_proposal: true, note: $("c-note").value || null, scope: $("c-scope").value });
    $("c-feedback").textContent = "Proposal saved as a good example (pending, not in any trained model).";
    $("c-note").value = ""; await renderClinician();
  } catch (e) { $("c-feedback").textContent = `Not saved: ${e.message}`; }
};

// ------------------------------------------------------------------ models & results
async function loadModels() {
  const m = await api("/models");
  $("m-versions").innerHTML = m.versions.map((v) => `<label class="opt">
    <input type="radio" name="mver" value="${esc(v.id)}" ${v.id === m.active ? "checked" : ""} ${v.available ? "" : "disabled"}>
    <span><b>${esc(v.id)}</b>${v.id === m.active ? ' <span class="chip">active for new sessions</span>' : ""}<br>
    <span class="small muted">${esc(v.label)}${v.available ? "" : ` · unavailable: ${esc(v.unavailable_reason)}`}</span></span></label>`).join("");
  const runs = (await api("/runs")).filter((r) => !r.smoke);
  $("m-run").innerHTML = runs.map((r) => `<option value="${esc(r.run_id)}">${esc(r.run_id)} (${esc(r.status)})</option>`).join("");
  if (runs.length) renderRun(runs[runs.length - 1].run_id);
}
$("m-run").onchange = () => renderRun($("m-run").value);

$("m-activate").onclick = async () => {
  const v = document.querySelector('input[name="mver"]:checked')?.value; if (!v) return;
  $("m-feedback").textContent = "Loading the model…";
  try { await post("/models/active", { version: v }); $("m-feedback").textContent = `New sessions will use ${v}.`; $("start-model").textContent = v; loadModels(); }
  catch (e) { $("m-feedback").textContent = e.message; }
};
$("m-session").onclick = async () => {
  const v = document.querySelector('input[name="mver"]:checked')?.value; if (!v) return;
  if (!P.sid) { $("m-feedback").textContent = "No practice session is open in this window."; return; }
  $("m-feedback").textContent = "Loading the model…";
  try { const r = await post(`/practice/sessions/${P.sid}/model`, { version: v }); renderSession(r); $("m-feedback").textContent = `Session ${P.sid} now uses ${v} for its next decisions.`; }
  catch (e) { $("m-feedback").textContent = e.message; }
};

async function renderRun(id) {
  $("m-run").value = id;
  const { manifest: m, evaluations } = await api(`/runs/${encodeURIComponent(id)}`);
  const rounds = (m.rounds || []).map((r) => `<tr><td>${r.round}</td><td>${r.samples_matching_demonstrator}/${r.samples}</td><td>${r.samples_unparseable}</td>
    <td>${r.pairs.used.expert} / ${r.pairs.used.replay} / ${r.pairs.used.intermodel}</td>
    <td>${r.pairs.available.expert} / ${r.pairs.available.replay} / ${r.pairs.available.intermodel}</td>
    <td>${r.steps}</td><td>${r.loss.length ? r.loss[0].toFixed(3) + " → " + r.loss[r.loss.length - 1].toFixed(3) : "–"}</td></tr>`).join("");
  $("m-run-detail").innerHTML = `<p><b>${esc(m.run_id)}</b> · status <b>${esc(m.status)}</b> · ${esc(m.method)}<br>
    <span class="small">base <code>${esc(m.base_model)}@${esc((m.base_revision || "").slice(0, 7))}</code> · device ${esc(m.device)} ·
    ${m.seconds ? m.seconds.toFixed(0) + " s" : ""} · seed ${m.seed} · dataset hash <code>${esc(m.demonstrations.hash)}</code> · finished ${esc(m.finished || "–")}</span></p>
    <div class="notice">This checkpoint was trained on the ${m.demonstrations.n} demonstrations listed below.
      Examples saved in the clinician view after ${esc(m.finished || m.started)} are <b>not</b> in it; they stay pending until a future run.</div>
    <details><summary class="small">Demonstrations used (${m.demonstrations.n})</summary><p class="small mono">${m.demonstrations.example_ids.map(esc).join(", ")}</p></details>
    <p class="small">SFT: ${m.sft ? m.sft.steps + " steps, loss " + m.sft.loss[0] + " → " + m.sft.loss[m.sft.loss.length - 1] : "–"}</p>
    <div class="tbl-wrap"><table class="tbl"><thead><tr><th>round</th><th>samples already matching</th><th>unparseable</th>
    <th>pairs used (expert / replay / intermodel)</th><th>available</th><th>DPO steps</th><th>loss</th></tr></thead><tbody>${rounds}</tbody></table></div>`;
  const ev = evaluations.test;
  if (!ev) { $("m-eval").innerHTML = '<span class="muted">No evaluation for this run yet. Run <code>uv run --extra llm python -m evaluation.compare --run ' + esc(id) + "</code>.</span>"; return; }
  const conds = Object.keys(ev.results);
  const head = `<p class="small">${ev.example_ids.length} reserved test situations, never used for training or prompts (hash <code>${esc(ev.dataset_hash)}</code>), computed ${esc(ev.created.slice(0, 16))}.
    Mock preferences of a fictional demonstrator; a small exploratory check, not a statistical or clinical validation.</p>`;
  const f = (v, n) => (v === null || v === undefined ? "n/a" : `${v}/${n}`);
  const table = `<div class="tbl-wrap"><table class="tbl"><thead><tr><th>model</th><th>valid output</th><th>constraint violations</th><th>fallbacks</th>
    <th>agrees (valid outputs)</th><th>agrees end to end</th><th>median latency</th></tr></thead><tbody>${conds.map((c) => {
      const r = ev.results[c];
      return `<tr><td><b>${esc(c)}</b></td><td>${f(r.valid_output, r.n)}</td><td>${f(r.constraint_violations, r.n)}</td><td>${f(r.fallbacks, r.n)}</td>
        <td>${r.agree_on_valid === null ? "n/a" : `${r.agree_on_valid}/${r.valid_output}`}</td><td><b>${r.agree_end_to_end}/${r.n}</b></td><td>${r.latency_ms_median.toFixed(0)} ms</td></tr>`; }).join("")}</tbody></table></div>`;
  const per = `<h3>Per situation</h3><p class="small muted">✓ among the demonstrator's accepted actions · ✗ not · † fallback to the bandit</p>
    <div class="tbl-wrap"><table class="tbl"><thead><tr><th>situation</th><th>rule</th><th>accepted</th>${conds.map((c) => `<th>${esc(c)}</th>`).join("")}</tr></thead><tbody>
    ${ev.example_ids.map((eid, i) => {
      const first = ev.results[conds[0]].per_example[i];
      const acc = first.accepted.map((k) => shortKey(k)).join(" or ");
      return `<tr><td class="small">${esc(eid.replace("demoA-v1-", ""))}</td><td>${esc(first.rule)}</td><td class="small">${esc(acc)}</td>${conds.map((c) => {
        const p = ev.results[c].per_example[i];
        return `<td class="small" title="${esc(p.raw_output || "")}">${esc(shortKey(p.final_action))} <span class="${p.agree_end_to_end ? "ok" : "miss"}">${p.agree_end_to_end ? "✓" : "✗"}</span>${p.fallback_reason ? "†" : ""}</td>`; }).join("")}</tr>`; }).join("")}
    </tbody></table></div>`;
  const dv = evaluations.dev;
  const dev = dv ? `<h3>Development split (${dv.example_ids.length} situations)</h3><p class="small">Agrees end to end: ${Object.entries(dv.results)
    .map(([c, r]) => `<b>${esc(c)}</b> ${r.agree_end_to_end}/${r.n}`).join(" · ")}. Read the test numbers together with these: with so few
    situations, one or two decisions change the picture.</p>` : "";
  $("m-eval").innerHTML = head + table + dev + per;
}
function shortKey(k) { const a = JSON.parse(k); return a.op === "retry" ? `retry ${HINT_LABEL[a.hint]}` : `next t${a.tier}`; }

initPractice();
