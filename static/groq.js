const stateSelect = document.getElementById("stateSelect");
const districtSelect = document.getElementById("districtSelect");
const blockSelect = document.getElementById("blockSelect");
const villageSelect = document.getElementById("villageSelect");
const yearSelect = document.getElementById("yearSelect");
const compareDistrict = document.getElementById("compareDistrict");
const compareYear = document.getElementById("compareYear");

const runGroqBtn = document.getElementById("runGroqBtn");
const exportPdfBtn = document.getElementById("exportPdfBtn");
const statusText = document.getElementById("statusText");
const darkToggle = document.getElementById("darkToggle");

const kpiRisk = document.getElementById("kpiRisk");
const kpiWqi = document.getElementById("kpiWqi");
const kpiHhi = document.getElementById("kpiHhi");
const kpiCompare = document.getElementById("kpiCompare");

const riskTrendChart = document.getElementById("riskTrendChart");
const scenarioDeltaChart = document.getElementById("scenarioDeltaChart");

const strategyTableBody = document.getElementById("strategyTableBody");
const analyticsTableBody = document.getElementById("analyticsTableBody");

const insightPanel = document.getElementById("insightPanel");
const scenarioPanel = document.getElementById("scenarioPanel");
const uncertaintyPanel = document.getElementById("uncertaintyPanel");
const comparisonPanel = document.getElementById("comparisonPanel");
const journalPanel = document.getElementById("journalPanel");

const chatMessages = document.getElementById("chatMessages");
const chatInput = document.getElementById("chatInput");
const chatSendBtn = document.getElementById("chatSendBtn");

const DARK_KEY = "st-wqhrnet-dark";
let latestBundle = null;

function setDarkMode(enabled) {
  const root = document.documentElement;
  if (enabled) {
    root.classList.add("dark");
    if (darkToggle) darkToggle.textContent = "Light";
    localStorage.setItem(DARK_KEY, "1");
  } else {
    root.classList.remove("dark");
    if (darkToggle) darkToggle.textContent = "Dark";
    localStorage.setItem(DARK_KEY, "0");
  }
}

function populateSelect(select, options) {
  if (!select) return;
  select.innerHTML = "";
  options.forEach((opt) => {
    const option = document.createElement("option");
    option.value = opt;
    option.textContent = opt;
    select.appendChild(option);
  });
}

function initYearOptions() {
  const years = [];
  for (let y = 2025; y <= 2050; y += 1) {
    years.push(String(y));
  }
  populateSelect(yearSelect, years);
  populateSelect(compareYear, years);
}

async function loadLocations() {
  const res = await fetch("/api/locations");
  const data = await res.json();
  const states = data.states || ["Tamil Nadu"];
  const districts = data.districts || [];

  populateSelect(stateSelect, states);
  populateSelect(districtSelect, districts);
  populateSelect(compareDistrict, districts);
}

async function loadBlocks(district) {
  if (!district) return;
  const res = await fetch(`/api/blocks?district=${encodeURIComponent(district)}`);
  const data = await res.json();
  const blocks = data.blocks || [];

  if (blocks.length === 0) {
    populateSelect(blockSelect, ["No Blocks"]);
    blockSelect.disabled = true;
    await loadVillages(district, "");
  } else {
    populateSelect(blockSelect, blocks);
    blockSelect.disabled = false;
    await loadVillages(district, blocks[0]);
  }
}

async function loadVillages(district, block) {
  if (!district) return;
  const res = await fetch(`/api/villages?district=${encodeURIComponent(district)}&block=${encodeURIComponent(block || "")}`);
  const data = await res.json();
  const villages = data.villages || [];

  if (villages.length === 0) {
    populateSelect(villageSelect, ["No Villages"]);
    villageSelect.disabled = true;
  } else {
    populateSelect(villageSelect, villages);
    villageSelect.disabled = false;
  }
}

function toPretty(obj) {
  if (!obj) return "No content.";
  try {
    return JSON.stringify(obj, null, 2);
  } catch (e) {
    return String(obj);
  }
}

function chartColors() {
  const dark = document.documentElement.classList.contains("dark");
  return {
    axis: dark ? "rgba(148,163,184,0.45)" : "rgba(15,23,42,0.2)",
    label: dark ? "rgba(226,232,240,0.9)" : "rgba(15,23,42,0.75)",
    base: "#2563eb",
    scenario: "#f97316",
    compare: "#14b8a6",
    risk: "#dc2626",
  };
}

function drawLineChart(canvas, labels, seriesA, seriesB) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = (canvas.width = canvas.offsetWidth);
  const h = (canvas.height = canvas.offsetHeight);
  const colors = chartColors();

  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = colors.axis;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(36, 14);
  ctx.lineTo(36, h - 26);
  ctx.lineTo(w - 10, h - 26);
  ctx.stroke();

  if (!labels.length) return;

  const values = [...seriesA, ...seriesB];
  const minVal = Math.min(...values);
  const maxVal = Math.max(...values);
  const span = maxVal - minVal || 1;
  const xStep = (w - 50) / (labels.length - 1 || 1);

  function drawSeries(arr, color) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.3;
    ctx.beginPath();
    arr.forEach((v, i) => {
      const x = 36 + i * xStep;
      const y = h - 26 - ((v - minVal) / span) * (h - 46);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    ctx.fillStyle = color;
    arr.forEach((v, i) => {
      const x = 36 + i * xStep;
      const y = h - 26 - ((v - minVal) / span) * (h - 46);
      ctx.beginPath();
      ctx.arc(x, y, 2.8, 0, Math.PI * 2);
      ctx.fill();
    });
  }

  drawSeries(seriesA, colors.base);
  drawSeries(seriesB, colors.scenario);

  ctx.fillStyle = colors.label;
  ctx.font = "10px Inter, system-ui, sans-serif";
  ctx.fillText(String(labels[0]), 36, h - 10);
  ctx.fillText(String(labels[labels.length - 1]), w - 34, h - 10);

  ctx.fillStyle = colors.base;
  ctx.fillRect(42, 16, 10, 3);
  ctx.fillStyle = colors.label;
  ctx.fillText("Baseline", 56, 20);
  ctx.fillStyle = colors.scenario;
  ctx.fillRect(120, 16, 10, 3);
  ctx.fillStyle = colors.label;
  ctx.fillText("Scenario", 136, 20);
}

function drawDeltaBars(canvas, labels, values) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = (canvas.width = canvas.offsetWidth);
  const h = (canvas.height = canvas.offsetHeight);
  const colors = chartColors();

  ctx.clearRect(0, 0, w, h);
  const maxAbs = Math.max(...values.map((v) => Math.abs(v)), 1);
  const xStep = (w - 40) / labels.length;
  const zeroY = h / 2;

  ctx.strokeStyle = colors.axis;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(26, 18);
  ctx.lineTo(26, h - 18);
  ctx.moveTo(26, zeroY);
  ctx.lineTo(w - 10, zeroY);
  ctx.stroke();

  labels.forEach((label, i) => {
    const x = 32 + i * xStep;
    const barW = Math.min(34, xStep - 8);
    const value = values[i];
    const barH = (Math.abs(value) / maxAbs) * (h * 0.34);

    ctx.fillStyle = value >= 0 ? colors.risk : colors.compare;
    const y = value >= 0 ? zeroY - barH : zeroY;
    ctx.fillRect(x, y, barW, barH);

    ctx.fillStyle = colors.label;
    ctx.font = "10px Inter, system-ui, sans-serif";
    ctx.fillText(label, x, h - 6);
    ctx.fillText(value.toFixed(2), x, value >= 0 ? y - 4 : y + barH + 12);
  });
}

function renderKpis(bundle) {
  const base = bundle.base_forecast || {};
  const analytics = bundle.analytics || {};
  const k = analytics.kpis || {};

  const wqi = base.wqi || {};
  const hhi = base.hhi || {};

  kpiRisk.textContent = `Risk Index: ${k.base_risk_index ?? "--"}`;
  kpiWqi.textContent = `WQI: ${wqi.value ?? "--"} (${wqi.class ?? "--"})`;
  kpiHhi.textContent = `HHI: ${hhi.value ?? "--"} (${hhi.class ?? "--"})`;
  kpiCompare.textContent = `Compare Delta: ${k.comparison_risk_delta ?? "--"}`;
}

function renderCharts(bundle) {
  const analytics = bundle.analytics || {};
  const horizon = analytics.horizon || [];

  const labels = horizon.map((x) => x.year);
  const baselineRisk = horizon.map((x) => Number(x.baseline_risk || 0));
  const scenarioRisk = horizon.map((x) => Number(x.scenario_risk || 0));
  drawLineChart(riskTrendChart, labels, baselineRisk, scenarioRisk);

  const k = analytics.kpis || {};
  const deltaLabels = ["Scenario Risk", "Compare Risk", "WQI Delta", "HHI Delta"];
  const deltaValues = [
    Number(k.scenario_risk_delta || 0),
    Number(k.comparison_risk_delta || 0),
    Number((bundle.management_plan || {}).scenario_delta?.wqi_delta || 0),
    Number((bundle.management_plan || {}).scenario_delta?.hhi_delta || 0),
  ];
  drawDeltaBars(scenarioDeltaChart, deltaLabels, deltaValues);
}

function renderStrategyTable(bundle) {
  const strategies = (bundle.management_plan || {}).strategies || [];
  strategyTableBody.innerHTML = "";

  strategies.forEach((s) => {
    const tr = document.createElement("tr");
    tr.className = "border-t border-slate-200";
    tr.innerHTML = `
      <td class="px-3 py-2"><span class="font-semibold">${s.priority}</span> (${s.priority_score})</td>
      <td class="px-3 py-2">${s.action}</td>
      <td class="px-3 py-2">${s.timeline}</td>
      <td class="px-3 py-2">${s.owner}</td>
      <td class="px-3 py-2">${s.expected_wqi_gain}</td>
      <td class="px-3 py-2">${s.expected_hhi_reduction}</td>
      <td class="px-3 py-2">${s.cost_band}</td>
    `;
    strategyTableBody.appendChild(tr);
  });

  if (!strategies.length) {
    strategyTableBody.innerHTML = '<tr><td class="px-3 py-3" colspan="7">No strategies available.</td></tr>';
  }
}

function renderAnalyticsTable(bundle) {
  const matrix = (bundle.analytics || {}).risk_matrix || [];
  analyticsTableBody.innerHTML = "";

  matrix.forEach((r) => {
    const tr = document.createElement("tr");
    tr.className = "border-t border-slate-200";
    tr.innerHTML = `
      <td class="px-3 py-2">${r.dimension}</td>
      <td class="px-3 py-2">${r.likelihood}</td>
      <td class="px-3 py-2">${r.impact}</td>
      <td class="px-3 py-2">${r.zone}</td>
    `;
    analyticsTableBody.appendChild(tr);
  });

  if (!matrix.length) {
    analyticsTableBody.innerHTML = '<tr><td class="px-3 py-3" colspan="4">No analytics available.</td></tr>';
  }
}

function renderNarratives(bundle) {
  const ai = bundle.ai || {};
  insightPanel.textContent = toPretty(ai.insight_panel || {});
  scenarioPanel.textContent = toPretty(ai.scenario_analysis || {});
  uncertaintyPanel.textContent = toPretty(ai.uncertainty_narrative || {});
  comparisonPanel.textContent = toPretty(ai.district_comparison || {});
  journalPanel.textContent = toPretty({
    management: ai.management_strategies || {},
    journal: ai.journal_summary || {},
  });
}

function renderBundle(bundle) {
  latestBundle = bundle;
  renderKpis(bundle);
  renderCharts(bundle);
  renderStrategyTable(bundle);
  renderAnalyticsTable(bundle);
  renderNarratives(bundle);
  if (exportPdfBtn) exportPdfBtn.disabled = false;
}

function appendChat(role, text, provider = "") {
  const row = document.createElement("div");
  row.className = role === "user" ? "mb-2 text-right" : "mb-2 text-left";
  const bubble = document.createElement("div");
  bubble.className = role === "user"
    ? "inline-block max-w-[90%] rounded-xl bg-primary-600 px-3 py-2 text-xs text-white"
    : "inline-block max-w-[90%] rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs text-slate-700";

  bubble.textContent = text;
  row.appendChild(bubble);
  if (provider && role !== "user") {
    const meta = document.createElement("div");
    meta.className = "mt-1 text-[10px] text-slate-500";
    meta.textContent = provider;
    row.appendChild(meta);
  }
  chatMessages.appendChild(row);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

async function checkGroqStatus() {
  const res = await fetch("/api/groq/status");
  const data = await res.json();
  if (!data.enabled) {
    statusText.textContent = "Groq unavailable. Advanced local fallback remains active.";
  } else {
    statusText.textContent = `Groq ready: ${data.model}`;
  }
}

function currentSelection() {
  return {
    state: stateSelect.value,
    district: districtSelect.value,
    block: blockSelect.disabled ? "" : blockSelect.value,
    village: villageSelect.disabled ? "" : villageSelect.value,
    year: Number(yearSelect.value),
  };
}

async function exportPdfAppendix() {
  if (!latestBundle) {
    statusText.textContent = "Generate analytics first, then export PDF.";
    return;
  }

  if (exportPdfBtn) {
    exportPdfBtn.disabled = true;
    exportPdfBtn.textContent = "Exporting...";
  }

  try {
    const res = await fetch("/api/ai/research_bundle/pdf", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ bundle: latestBundle, selection: currentSelection() }),
    });

    if (!res.ok) {
      let errMsg = `PDF export failed (${res.status})`;
      try {
        const errJson = await res.json();
        errMsg = errJson.error || errMsg;
      } catch (_) {
        // keep default message
      }
      statusText.textContent = errMsg;
      return;
    }

    const blob = await res.blob();
    const cd = res.headers.get("Content-Disposition") || "";
    const m = cd.match(/filename="?([^";]+)"?/i);
    const filename = m && m[1] ? m[1] : "research_bundle.pdf";

    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
    statusText.textContent = "PDF appendix downloaded.";
  } catch (err) {
    statusText.textContent = `PDF export error: ${err.message}`;
  } finally {
    if (exportPdfBtn) {
      exportPdfBtn.disabled = false;
      exportPdfBtn.textContent = "Export PDF Appendix";
    }
  }
}

function payloadForBundle() {
  return {
    ...currentSelection(),
    scenario: {
      rainfall_change_pct: Number(document.getElementById("scRain").value || 0),
      temperature_change_c: Number(document.getElementById("scTemp").value || 0),
      agri_change_pct: Number(document.getElementById("scAgri").value || 0),
      population_change_pct: Number(document.getElementById("scPop").value || 0),
    },
    compare: {
      district: compareDistrict.value,
      year: Number(compareYear.value),
    },
  };
}

async function runResearchBundle() {
  runGroqBtn.disabled = true;
  if (exportPdfBtn) exportPdfBtn.disabled = true;
  runGroqBtn.textContent = "Generating...";
  statusText.textContent = "Running forecasts, management synthesis, and AI generation...";

  try {
    const res = await fetch("/api/ai/research_bundle", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payloadForBundle()),
    });
    const data = await res.json();

    if (!data.ok) {
      statusText.textContent = data.error || "Research generation failed.";
      return;
    }

    renderBundle(data);
    if (data.warning) {
      statusText.textContent = data.warning;
    } else {
      statusText.textContent = "Advanced analytics and management intelligence generated.";
    }

    appendChat("assistant", "Research bundle is ready. Ask me about strategy trade-offs, confidence, or operational priorities.", "system");
  } catch (err) {
    statusText.textContent = `Error: ${err.message}`;
  } finally {
    runGroqBtn.disabled = false;
    runGroqBtn.textContent = "Generate Advanced Analytics";
  }
}

function buildChatContext() {
  if (!latestBundle) return payloadForBundle();
  return {
    base_forecast: latestBundle.base_forecast || {},
    scenario_forecast: latestBundle.scenario_forecast || {},
    comparison_forecast: latestBundle.comparison_forecast || {},
    management_plan: latestBundle.management_plan || {},
    analytics: latestBundle.analytics || {},
    ai: latestBundle.ai || {},
  };
}

async function sendChat() {
  const q = (chatInput.value || "").trim();
  if (!q) return;
  appendChat("user", q);
  chatInput.value = "";

  try {
    const res = await fetch("/api/ai/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q, context: buildChatContext() }),
    });
    const data = await res.json();
    if (!data.ok) {
      appendChat("assistant", data.error || "Chat request failed.", "error");
      return;
    }
    appendChat("assistant", data.answer || "No answer generated.", data.provider || "assistant");
  } catch (err) {
    appendChat("assistant", `Chat error: ${err.message}`, "error");
  }
}

const savedDark = localStorage.getItem(DARK_KEY);
if (savedDark === "1") {
  setDarkMode(true);
}
if (darkToggle) {
  darkToggle.addEventListener("click", () => {
    setDarkMode(!document.documentElement.classList.contains("dark"));
    if (latestBundle) renderCharts(latestBundle);
  });
}

districtSelect.addEventListener("change", () => loadBlocks(districtSelect.value));
blockSelect.addEventListener("change", () => loadVillages(districtSelect.value, blockSelect.value));
runGroqBtn.addEventListener("click", runResearchBundle);
if (exportPdfBtn) exportPdfBtn.addEventListener("click", exportPdfAppendix);
chatSendBtn.addEventListener("click", sendChat);
chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    sendChat();
  }
});

initYearOptions();
loadLocations().then(() => loadBlocks(districtSelect.value));
checkGroqStatus();
appendChat("assistant", "Groq analyst chatbot ready. Generate analytics first for context-rich answers.", "system");
