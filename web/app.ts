/** Dashboard: fetches the snapshot once, then ranks client-side with core on every control change. */
import { applePresets, efficiencyByPreset, rankModels, type Calibration, type Ranked, type Snapshot } from "../src/core";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const inputs = { mem: $<HTMLInputElement>("mem"), bw: $<HTMLInputElement>("bw"), ctx: $<HTMLSelectElement>("ctx"), tps: $<HTMLInputElement>("tps"), eff: $<HTMLInputElement>("eff") };
const preset = $<HTMLSelectElement>("preset");
const TOP = 50;
let snap: Snapshot | undefined;
let measured: Record<string, number> = {};

const presets = applePresets();
preset.innerHTML = `<option value="">Manual</option>` + presets.map((p, i) => `<option value="${i}">${p.label} · ${p.bandwidthGBs} GB/s</option>`).join("");
preset.value = String(presets.findIndex((p) => p.label === "M4 Max (full GPU)"));
/** Selecting a preset sets its bandwidth, and its measured efficiency when a calibration exists. */
function applyPreset() {
  const p = presets[Number(preset.value)];
  if (!preset.value || !p) return;
  inputs.bw.value = String(p.bandwidthGBs);
  if (measured[p.label]) inputs.eff.value = measured[p.label].toFixed(2);
}
preset.onchange = () => { applyPreset(); render(); };
inputs.bw.oninput = () => { preset.value = ""; render(); };
for (const el of [inputs.mem, inputs.ctx, inputs.tps, inputs.eff]) el.oninput = render;

const esc = (s: string) => s.replace(/[&<>"]/g, (c) => `&#${c.charCodeAt(0)};`);
const fmtB = (n: number) => (n / 1e9 >= 10 ? (n / 1e9).toFixed(0) : (n / 1e9).toFixed(1)) + "B";

function render() {
  $("mem-out").textContent = `${inputs.mem.value} GB`;
  $("bw-out").textContent = `${inputs.bw.value} GB/s`;
  $("tps-out").textContent = `${inputs.tps.value} tok/s`;
  const m = measured[presets[Number(preset.value)]?.label ?? ""];
  $("eff-out").textContent = Number(inputs.eff.value).toFixed(2) + (preset.value && m ? ` (measured ${m.toFixed(2)})` : "");
  if (!snap) return;
  const hw = { memBudgetGB: Number(inputs.mem.value), bandwidthGBs: Number(inputs.bw.value), label: preset.selectedOptions[0]?.textContent ?? "" };
  const o = { ctx: Number(inputs.ctx.value), minTps: Number(inputs.tps.value), efficiency: Number(inputs.eff.value) };
  const { rows, rejects } = rankModels(snap.models, snap.aa, hw, o);
  $("summary").textContent = `${rows.length} of ${snap.models.length} models run · rejected ${rejects["too big"]} too big, ${rejects["too slow"]} too slow` +
    (snap.aa.length ? "" : " · no AA scores in snapshot: order is by size, NOT intelligence");
  $("rows").innerHTML = rows.slice(0, TOP).map((r, i) => `<tr><td>${i + 1}</td>` +
    `<td><a href="https://huggingface.co/${esc(r.id)}">${esc(r.id)}</a></td><td>${fmtB(r.total)} (${fmtB(r.active)} act)</td><td>${r.quant.name}</td>` +
    `<td class="num">${r.memGB.toFixed(1)}</td><td class="num">${Math.round(r.tps)}</td>` +
    `<td class="num" title="${esc(r.aa?.name ?? "")}">${r.aa ? r.aa.score.toFixed(1) : "—"}</td><td>${esc(r.created)}</td></tr>`).join("");
  chart(rows.filter((r) => r.aa));
}

/** Score vs est tok/s (log x); hover shows the model. */
function chart(rows: Ranked[]) {
  const W = 800, H = 320, L = 44, R = 16, T = 16, B = 32;
  const svg = $("chart");
  if (!rows.length) { svg.innerHTML = `<text x="${W / 2}" y="${H / 2}" text-anchor="middle">No scored models fit these settings</text>`; return; }
  const xs = rows.map((r) => Math.log10(r.tps)), ys = rows.map((r) => r.aa!.score);
  const x0 = Math.floor(Math.min(...xs) * 2) / 2, x1 = Math.ceil(Math.max(...xs) * 2) / 2 || x0 + 0.5;
  const y0 = Math.floor(Math.min(...ys) / 10) * 10, y1 = Math.ceil(Math.max(...ys) / 10) * 10 || y0 + 10;
  const sx = (v: number) => L + ((v - x0) / (x1 - x0 || 1)) * (W - L - R);
  const sy = (v: number) => H - B - ((v - y0) / (y1 - y0 || 1)) * (H - T - B);
  let g = `<line class="axis" x1="${L}" y1="${H - B}" x2="${W - R}" y2="${H - B}"/><line class="axis" x1="${L}" y1="${T}" x2="${L}" y2="${H - B}"/>`;
  for (const t of [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]) {
    const lx = Math.log10(t);
    if (lx >= x0 && lx <= x1) g += `<text x="${sx(lx)}" y="${H - B + 16}" text-anchor="middle">${t}</text>`;
  }
  for (let v = y0; v <= y1; v += 10) g += `<text x="${L - 6}" y="${sy(v) + 4}" text-anchor="end">${v}</text>`;
  g += `<text x="${(L + W - R) / 2}" y="${H - 2}" text-anchor="middle">est tok/s (log)</text>`;
  g += `<text x="12" y="${(T + H - B) / 2}" transform="rotate(-90 12 ${(T + H - B) / 2})" text-anchor="middle">Intelligence Index</text>`;
  g += rows.map((r) => `<circle cx="${sx(Math.log10(r.tps))}" cy="${sy(r.aa!.score)}" r="4"><title>${esc(r.id)} · ${r.quant.name} · ${Math.round(r.tps)} tok/s · ${r.aa!.score.toFixed(1)}</title></circle>`).join("");
  svg.innerHTML = g;
}

render();
fetch("/api/calibration").then((r) => (r.ok ? r.json() : [])).then((cals: Calibration[]) => { measured = efficiencyByPreset(cals); applyPreset(); render(); }).catch(() => {});
fetch("/api/snapshot").then(async (r) => {
  if (!r.ok) { $("status").textContent = `Snapshot unavailable: ${r.status} ${await r.text()}`; return; }
  snap = await r.json() as Snapshot;
  $("status").textContent = `Snapshot ${new Date(snap.fetchedAt).toLocaleString()} · ${snap.models.length} models · ${snap.gated.length} gated · ${snap.errors.length} fetch errors`;
  render();
});
