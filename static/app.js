const stateSelect = document.getElementById("stateSelect");
const districtSelect = document.getElementById("districtSelect");
const blockSelect = document.getElementById("blockSelect");
const villageSelect = document.getElementById("villageSelect");
const yearSelect = document.getElementById("yearSelect");
const submitBtn = document.getElementById("submitBtn");

const wqiOutput = document.getElementById("wqiOutput");
const hhiOutput = document.getElementById("hhiOutput");
const locationOutput = document.getElementById("locationOutput");
const tipLocation = document.getElementById("tip-location");
const tipWqi = document.getElementById("tip-wqi");
const tipHhi = document.getElementById("tip-hhi");

const wqiChart = document.getElementById("wqiChart");
const hhiChart = document.getElementById("hhiChart");

const mapCanvas = document.getElementById("mapCanvas");
const mapHint = document.getElementById("mapHint");
const mapLabelToggle = document.getElementById("mapLabelToggle");
const refreshBtn = document.getElementById("refreshBtn");

const darkToggle = document.getElementById("darkToggle");
const DARK_KEY = "st-wqhrnet-dark";
const MAP_LABEL_KEY = "st-wqhrnet-map-labels";

const calcWqiBtn = document.getElementById("calcWqiBtn");
const calcHhiBtn = document.getElementById("calcHhiBtn");
const wqiCalcOutput = document.getElementById("wqiCalcOutput");
const hhiCalcOutput = document.getElementById("hhiCalcOutput");
const calcInputs = Array.from(
  document.querySelectorAll(
    "#wqiPH,#wqiDO,#wqiTDS,#wqiNO3,#wqiCl,#wqiF,#hhiAs,#hhiPb,#hhiCd,#hhiIR,#hhiEF,#hhiED,#hhiBW,#hhiAT,#hhiRfDAs,#hhiRfDPb,#hhiRfDCd"
  )
);

const DISTRICT_ALIASES = {
  sivagangai: "sivaganga",
  vilupuram: "viluppuram",
  tiruvallur: "thiruvallur",
  tiruvarur: "thiruvarur",
};

const PREDICTION_DELAY_MIN_MS = 10000;
const PREDICTION_DELAY_MAX_MS = 15000;
let states = [];
let districts = [];
let latestTrend = null;
let latestPrediction = null;

let districtMap = null;
let districtLayerGroup = null;
let districtLabelLayerGroup = null;
let districtLayerByKey = new Map();
let labelsVisible = localStorage.getItem(MAP_LABEL_KEY) === "1";

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

function normalizeDistrictName(name) {
  const key = String(name || "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "");
  return DISTRICT_ALIASES[key] || key;
}

function currentPredictionForDistrict(district) {
  if (!latestPrediction) return null;
  const a = normalizeDistrictName(latestPrediction.district || "");
  const b = normalizeDistrictName(district || "");
  return a === b ? latestPrediction : null;
}

function neutralDistrictStyle() {
  return {
    color: "#94a3b8",
    weight: 1.2,
    fillColor: "#e2e8f0",
    fillOpacity: 0.62,
    opacity: 0.95,
  };
}

function hoverDistrictStyle() {
  return {
    color: "#64748b",
    weight: 1.8,
    fillColor: "#cbd5e1",
    fillOpacity: 0.72,
    opacity: 1,
  };
}

function selectedDistrictStyle(result) {
  const riskColorByWqi = {
    Excellent: "#22c55e",
    Good: "#84cc16",
    Poor: "#f59e0b",
    "Very Poor": "#f97316",
    "Unsuitable for Drinking": "#dc2626",
  };
  const fallbackColor = "#2563eb";
  const color = result && result.wqi ? riskColorByWqi[result.wqi.class] || fallbackColor : fallbackColor;

  return {
    color,
    weight: 2.8,
    fillColor: color,
    fillOpacity: 0.58,
    opacity: 1,
  };
}

function popupHTML(district, result) {
  const wqiText = result && result.wqi ? `${result.wqi.value} (${result.wqi.class})` : "--";
  const hhiText = result && result.hhi ? `${result.hhi.value} (${result.hhi.class})` : "--";
  return `
    <div class="map-popup-title">${district}</div>
    <div class="map-popup-row"><span>WQI</span><strong>${wqiText}</strong></div>
    <div class="map-popup-row"><span>HHI</span><strong>${hhiText}</strong></div>
  `;
}

function updateHint(text) {
  if (mapHint) mapHint.textContent = text;
}

function setMapLabelsVisible(visible) {
  labelsVisible = Boolean(visible);
  localStorage.setItem(MAP_LABEL_KEY, labelsVisible ? "1" : "0");

  if (mapLabelToggle) {
    mapLabelToggle.textContent = labelsVisible ? "Labels On" : "Labels Off";
    mapLabelToggle.setAttribute("aria-pressed", labelsVisible ? "true" : "false");
  }

  if (!districtMap || !districtLabelLayerGroup) return;

  if (labelsVisible) {
    if (!districtMap.hasLayer(districtLabelLayerGroup)) {
      districtLabelLayerGroup.addTo(districtMap);
    }
  } else if (districtMap.hasLayer(districtLabelLayerGroup)) {
    districtMap.removeLayer(districtLabelLayerGroup);
  }
}

function getLayerForDistrict(district) {
  return districtLayerByKey.get(normalizeDistrictName(district));
}

function refreshDistrictStyles(selectedDistrict, selectedResult) {
  if (!districtLayerGroup) return;
  districtLayerGroup.eachLayer((layer) => {
    const layerDistrict = layer.feature?.properties?.district || "";
    const isSelected = normalizeDistrictName(layerDistrict) === normalizeDistrictName(selectedDistrict || "");
    layer.setStyle(isSelected ? selectedDistrictStyle(selectedResult) : neutralDistrictStyle());
  });
}

function highlightDistrictOnMap(district, result = null, options = {}) {
  if (!districtMap || !districtLayerGroup || !district) return;
  const { zoom = true, popup = true } = options;
  const layer = getLayerForDistrict(district);
  if (!layer) return;

  refreshDistrictStyles(district, result);
  layer.bringToFront();

  if (zoom) {
    districtMap.fitBounds(layer.getBounds(), { padding: [24, 24], maxZoom: 9 });
  }

  if (popup) {
    layer.bindPopup(popupHTML(district, result), { closeButton: false, autoPanPadding: [18, 18] }).openPopup();
  }

  if (result) {
    updateHint(`Selected: ${district} | WQI ${result.wqi.value} (${result.wqi.class}) | HHI ${result.hhi.value} (${result.hhi.class})`);
  } else {
    updateHint(`Selected: ${district}. Click Submit to fetch forecast metrics.`);
  }
}

async function initDistrictMap() {
  if (!window.L || !mapCanvas) return;
  districtMap = L.map("tnLeafletMap", {
    zoomControl: true,
    attributionControl: false,
    minZoom: 6,
    maxZoom: 12,
  });

  districtMap.setView([11.1271, 78.6569], 7);

  const res = await fetch("/static/tn_districts.geojson");
  if (!res.ok) {
    updateHint("District boundary data not available.");
    return;
  }
  const geo = await res.json();

  districtLabelLayerGroup = L.layerGroup();

  districtLayerGroup = L.geoJSON(geo, {
    style: neutralDistrictStyle,
    onEachFeature: (feature, layer) => {
      const district = feature?.properties?.district || "Unknown";
      const key = normalizeDistrictName(district);
      districtLayerByKey.set(key, layer);

      layer.bindTooltip(district, {
        sticky: true,
        direction: "top",
        className: "district-tooltip",
        opacity: 0.95,
      });

      const center = layer.getBounds().getCenter();
      const labelMarker = L.marker(center, {
        interactive: false,
        icon: L.divIcon({
          className: "district-label-marker",
          html: district,
          iconSize: null,
        }),
      });
      districtLabelLayerGroup.addLayer(labelMarker);

      layer.on({
        mouseover: (e) => {
          const selected = districtSelect ? districtSelect.value : "";
          if (normalizeDistrictName(selected) !== key) {
            e.target.setStyle(hoverDistrictStyle());
          }
        },
        mouseout: () => {
          const selected = districtSelect ? districtSelect.value : "";
          const selectedRes = currentPredictionForDistrict(selected);
          refreshDistrictStyles(selected, selectedRes);
        },
        click: () => {
          if (districtSelect && districtSelect.value !== district) {
            districtSelect.value = district;
            districtSelect.dispatchEvent(new Event("change"));
          } else {
            highlightDistrictOnMap(district, currentPredictionForDistrict(district), {
              zoom: true,
              popup: true,
            });
          }
        },
      });
    },
  }).addTo(districtMap);

  const bounds = districtLayerGroup.getBounds();
  if (bounds && bounds.isValid()) {
    districtMap.fitBounds(bounds, { padding: [14, 14] });
  }
  setMapLabelsVisible(labelsVisible);
  updateHint("Hover districts for names. Select a district to zoom and view forecast.");
}

setMapLabelsVisible(labelsVisible);

const savedDark = localStorage.getItem(DARK_KEY);
if (savedDark === "1") {
  setDarkMode(true);
}
if (mapLabelToggle) {
  mapLabelToggle.addEventListener("click", () => {
    setMapLabelsVisible(!labelsVisible);
  });
}

if (darkToggle) {
  darkToggle.addEventListener("click", () => {
    setDarkMode(!document.documentElement.classList.contains("dark"));
    renderCharts();
    if (districtMap) {
      setTimeout(() => districtMap.invalidateSize(), 120);
    }
  });
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
}

async function loadLocations() {
  const res = await fetch("/api/locations");
  const data = await res.json();
  states = data.states || ["Tamil Nadu"];
  districts = data.districts || [];
  if (stateSelect) populateSelect(stateSelect, states);
  populateSelect(districtSelect, districts);
}

async function loadBlocks(district) {
  if (!district || !blockSelect) return;
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
  if (!district || !villageSelect) return;
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

async function logEvent(event, data) {
  try {
    await fetch("/api/log_event", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ event, data }),
    });
  } catch (e) {
    // best-effort only
  }
}

function drawChart(canvas, years, values, color) {
  const ctx = canvas.getContext("2d");
  const w = (canvas.width = canvas.offsetWidth);
  const h = (canvas.height = canvas.offsetHeight);
  const dark = document.documentElement.classList.contains("dark");

  const axisColor = dark ? "rgba(148, 163, 184, 0.45)" : "rgba(15, 23, 42, 0.18)";
  const labelColor = dark ? "rgba(226, 232, 240, 0.9)" : "rgba(15, 23, 42, 0.7)";

  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = axisColor;
  ctx.lineWidth = 1;

  ctx.beginPath();
  ctx.moveTo(30, 10);
  ctx.lineTo(30, h - 25);
  ctx.lineTo(w - 10, h - 25);
  ctx.stroke();

  if (years.length === 0) return;

  const minVal = Math.min(...values);
  const maxVal = Math.max(...values);
  const span = maxVal - minVal || 1;
  const xStep = (w - 50) / (years.length - 1 || 1);

  ctx.strokeStyle = color;
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  values.forEach((v, i) => {
    const x = 30 + i * xStep;
    const y = h - 30 - ((v - minVal) / span) * (h - 50);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  ctx.fillStyle = color;
  values.forEach((v, i) => {
    const x = 30 + i * xStep;
    const y = h - 30 - ((v - minVal) / span) * (h - 50);
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, Math.PI * 2);
    ctx.fill();
  });

  ctx.fillStyle = labelColor;
  ctx.font = "10px ui-sans-serif, system-ui, -apple-system, Segoe UI, Arial";
  ctx.fillText(years[0], 30, h - 10);
  ctx.fillText(years[years.length - 1], w - 35, h - 10);
}

function renderCharts() {
  if (!latestTrend) return;
  drawChart(wqiChart, latestTrend.years, latestTrend.wqiValues, "#2563eb");
  drawChart(hhiChart, latestTrend.years, latestTrend.hhiValues, "#0ea5e9");
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}


function predictionDelayMs() {
  return Math.floor(Math.random() * (PREDICTION_DELAY_MAX_MS - PREDICTION_DELAY_MIN_MS + 1)) + PREDICTION_DELAY_MIN_MS;
}

function formatRange(range) {
  return `${range[0].toFixed(2)} - ${range[1].toFixed(2)}`;
}

async function submitPrediction() {
  if (submitBtn) {
    submitBtn.disabled = true;
    submitBtn.dataset.prevText = submitBtn.textContent;
    submitBtn.textContent = "Predicting...";
  }

  const payload = {
    state: stateSelect ? stateSelect.value : "Tamil Nadu",
    district: districtSelect.value,
    block: blockSelect && !blockSelect.disabled ? blockSelect.value : "",
    village: villageSelect && !villageSelect.disabled ? villageSelect.value : "",
    year: yearSelect.value,
  };

  const res = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();

  if (!data.ok) {
    alert(data.error || "Prediction failed");
    if (submitBtn) {
      submitBtn.disabled = false;
      submitBtn.textContent = submitBtn.dataset.prevText || "Submit";
    }
    return;
  }

  const result = data.result;
  await sleep(predictionDelayMs());

  const wqi = result.wqi;
  const hhi = result.hhi;

  wqiOutput.textContent = `WQI: ${formatRange(wqi.range)} (${wqi.class})`;
  hhiOutput.textContent = `HHI: ${formatRange(hhi.range)} (${hhi.class})`;

  const blockLabel = result.block && result.block !== "No Blocks" ? `, ${result.block}` : "";
  const villageLabel = result.village && result.village !== "No Villages" ? `, ${result.village}` : "";
  if (tipLocation) tipLocation.textContent = `${result.district}${blockLabel}${villageLabel}`;
  if (tipWqi) tipWqi.textContent = `${wqi.value} (${wqi.class})`;
  if (tipHhi) tipHhi.textContent = `${hhi.value} (${hhi.class})`;

  if (locationOutput) {
    locationOutput.textContent = `Location: ${result.district}${blockLabel}${villageLabel}`;
  }

  latestPrediction = result;
  highlightDistrictOnMap(result.district, result, { zoom: true, popup: true });

  const years = result.trend.map((t) => t.year);
  const wqiValues = result.trend.map((t) => t.wqi_value);
  const hhiValues = result.trend.map((t) => t.hhi_value);
  latestTrend = { years, wqiValues, hhiValues };
  renderCharts();

  logEvent("predict_client", { ...payload });

  if (submitBtn) {
    submitBtn.disabled = false;
    submitBtn.textContent = submitBtn.dataset.prevText || "Submit";
  }
}

function safeNumber(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function clearCalcInputs() {
  calcInputs.forEach((input) => {
    input.value = "";
  });
  wqiCalcOutput.textContent = "Enter values to calculate WQI.";
  hhiCalcOutput.textContent = "Enter values to calculate HHI.";
}

function toggleCalcButtons() {
  const wqiIds = ["wqiPH", "wqiDO", "wqiTDS", "wqiNO3", "wqiCl", "wqiF"];
  const hhiIds = [
    "hhiAs",
    "hhiPb",
    "hhiCd",
    "hhiIR",
    "hhiEF",
    "hhiED",
    "hhiBW",
    "hhiAT",
    "hhiRfDAs",
    "hhiRfDPb",
    "hhiRfDCd",
  ];

  const wqiReady = wqiIds.every((id) => document.getElementById(id).value !== "");
  const hhiReady = hhiIds.every((id) => document.getElementById(id).value !== "");
  if (calcWqiBtn) calcWqiBtn.disabled = !wqiReady;
  if (calcHhiBtn) calcHhiBtn.disabled = !hhiReady;
}

function wqiClass(value) {
  if (value <= 25) return "Excellent";
  if (value <= 50) return "Good";
  if (value <= 75) return "Poor";
  if (value <= 100) return "Very Poor";
  return "Unsuitable for Drinking";
}

function hhiClass(value) {
  if (value < 1.0) return "Low Risk";
  if (Math.abs(value - 1.0) < 1e-6) return "Threshold Level";
  return "High Risk";
}

function rangeForClass(value, cls) {
  let lo = 0;
  let hi = 100;

  if (cls === "Excellent") [lo, hi] = [0, 25];
  if (cls === "Good") [lo, hi] = [26, 50];
  if (cls === "Poor") [lo, hi] = [51, 75];
  if (cls === "Very Poor") [lo, hi] = [76, 100];
  if (cls === "Unsuitable for Drinking") [lo, hi] = [100.01, 120];

  if (cls === "Low Risk") [lo, hi] = [0, 0.9999];
  if (cls === "Threshold Level") [lo, hi] = [0.9999, 1.0001];
  if (cls === "High Risk") [lo, hi] = [1.0001, 3];

  const width = (hi - lo) * 0.2;
  const vLo = Math.max(lo, value - width);
  const vHi = Math.min(hi, value + width);
  return [vLo, vHi];
}

function calcWQI() {
  const ph = safeNumber(document.getElementById("wqiPH").value);
  const doVal = safeNumber(document.getElementById("wqiDO").value);
  const tds = safeNumber(document.getElementById("wqiTDS").value);
  const no3 = safeNumber(document.getElementById("wqiNO3").value);
  const cl = safeNumber(document.getElementById("wqiCl").value);
  const f = safeNumber(document.getElementById("wqiF").value);

  if ([ph, doVal, tds, no3, cl, f].some((v) => v === null)) {
    wqiCalcOutput.textContent = "Please enter all WQI values.";
    return;
  }

  const params = [
    { name: "pH", v: ph, standard: 8.5, ideal: 7.0 },
    { name: "DO", v: doVal, standard: 5.0, ideal: 14.6 },
    { name: "TDS", v: tds, standard: 500.0, ideal: 0.0 },
    { name: "NO3", v: no3, standard: 45.0, ideal: 0.0 },
    { name: "Cl", v: cl, standard: 250.0, ideal: 0.0 },
    { name: "F", v: f, standard: 1.5, ideal: 0.0 },
  ];

  let sumQiWi = 0;
  let sumWi = 0;

  params.forEach((p) => {
    const qi = ((p.v - p.ideal) / (p.standard - p.ideal)) * 100;
    const wi = 1 / p.standard;
    sumQiWi += qi * wi;
    sumWi += wi;
  });

  const wqi = sumQiWi / sumWi;
  const cls = wqiClass(wqi);
  const range = rangeForClass(wqi, cls);

  wqiCalcOutput.textContent = `WQI: ${range[0].toFixed(2)} - ${range[1].toFixed(2)} - ${cls}`;
  logEvent("calc_wqi", {
    state: stateSelect ? stateSelect.value : "Tamil Nadu",
    district: districtSelect ? districtSelect.value : "",
    year: yearSelect ? yearSelect.value : "",
    inputs: { ph, do: doVal, tds, no3, cl, f },
    output: wqiCalcOutput.textContent,
  });
}

function calcHHI() {
  const asVal = safeNumber(document.getElementById("hhiAs").value);
  const pbVal = safeNumber(document.getElementById("hhiPb").value);
  const cdVal = safeNumber(document.getElementById("hhiCd").value);
  const IR = safeNumber(document.getElementById("hhiIR").value);
  const EF = safeNumber(document.getElementById("hhiEF").value);
  const ED = safeNumber(document.getElementById("hhiED").value);
  const BW = safeNumber(document.getElementById("hhiBW").value);
  const AT = safeNumber(document.getElementById("hhiAT").value);
  const rfdAs = safeNumber(document.getElementById("hhiRfDAs").value);
  const rfdPb = safeNumber(document.getElementById("hhiRfDPb").value);
  const rfdCd = safeNumber(document.getElementById("hhiRfDCd").value);

  if ([asVal, pbVal, cdVal, IR, EF, ED, BW, AT, rfdAs, rfdPb, rfdCd].some((v) => v === null)) {
    hhiCalcOutput.textContent = "Please enter all HHI values.";
    return;
  }

  function add(C, rfd) {
    const ADD = (C * IR * EF * ED) / (BW * AT);
    return ADD / rfd;
  }

  const hqAs = add(asVal, rfdAs);
  const hqPb = add(pbVal, rfdPb);
  const hqCd = add(cdVal, rfdCd);
  const hhi = hqAs + hqPb + hqCd;

  const cls = hhiClass(hhi);
  const range = rangeForClass(hhi, cls);

  hhiCalcOutput.textContent = `HHI: ${range[0].toFixed(2)} - ${range[1].toFixed(2)} - ${cls}`;
  logEvent("calc_hhi", {
    state: stateSelect ? stateSelect.value : "Tamil Nadu",
    district: districtSelect ? districtSelect.value : "",
    year: yearSelect ? yearSelect.value : "",
    inputs: { as: asVal, pb: pbVal, cd: cdVal, IR, EF, ED, BW, AT, rfdAs, rfdPb, rfdCd },
    output: hhiCalcOutput.textContent,
  });
}

if (districtSelect) {
  districtSelect.addEventListener("change", async () => {
    await loadBlocks(districtSelect.value);
    highlightDistrictOnMap(districtSelect.value, currentPredictionForDistrict(districtSelect.value), {
      zoom: true,
      popup: false,
    });
  });
}

if (stateSelect) {
  stateSelect.addEventListener("change", () => {
    logEvent("state_change", { state: stateSelect.value });
  });
}

if (blockSelect) {
  blockSelect.addEventListener("change", () => {
    loadVillages(districtSelect.value, blockSelect.value);
  });
}

if (submitBtn) {
  submitBtn.addEventListener("click", submitPrediction);
}
if (calcWqiBtn) {
  calcWqiBtn.addEventListener("click", calcWQI);
}
if (calcHhiBtn) {
  calcHhiBtn.addEventListener("click", calcHHI);
}
if (refreshBtn) {
  refreshBtn.addEventListener("click", () => {
    clearCalcInputs();
    window.location.reload();
  });
}

initYearOptions();
Promise.all([loadLocations(), initDistrictMap()]).then(async () => {
  const selectedDistrict = districtSelect ? districtSelect.value : "";
  if (selectedDistrict) {
    await loadBlocks(selectedDistrict);
    highlightDistrictOnMap(selectedDistrict, null, { zoom: true, popup: false });
  }
});

window.addEventListener("resize", () => {
  if (districtMap) {
    districtMap.invalidateSize();
  }
});

calcInputs.forEach((input) => {
  input.setAttribute("autocomplete", "off");
  input.addEventListener("input", toggleCalcButtons);
});
clearCalcInputs();
toggleCalcButtons();
