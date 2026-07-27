// MPFragment Studio - Desktop GUI Logic Engine

let selectedFile = null;
let lastResults = null;

document.addEventListener('DOMContentLoaded', () => {
  initUI();
  checkApiHealth();
});

async function checkApiHealth() {
  const statusText = document.getElementById('status-text');
  try {
    const res = await fetch('/api/health');
    if (res.ok) {
      const data = await res.json();
      if (data.model_exists) {
        statusText.textContent = "Engine Ready";
      } else {
        statusText.textContent = "ONNX Model Not Found";
      }
    }
  } catch (e) {
    console.warn("Health check error:", e);
  }
}


function initUI() {
  const fileInput = document.getElementById('file-input');
  const dropzone = document.getElementById('dropzone');
  const btnBrowse = document.getElementById('btn-browse-file');
  const btnRun = document.getElementById('btn-run-inference');
  const btnExportCsv = document.getElementById('btn-export-csv');
  const btnExportJson = document.getElementById('btn-export-json');
  const btnExportCsvTable = document.getElementById('btn-export-csv-table');

  // Sliders
  const scoreSlider = document.getElementById('score-thresh');
  const wbfSlider = document.getElementById('wbf-thresh');
  const tileSlider = document.getElementById('tile-size');

  scoreSlider.addEventListener('input', updateBadges);
  wbfSlider.addEventListener('input', updateBadges);
  tileSlider.addEventListener('input', updateBadges);
  updateBadges();

  // Click triggers for File Selection
  dropzone.addEventListener('click', () => fileInput.click());
  btnBrowse.addEventListener('click', () => fileInput.click());

  // Native File Input change
  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });

  // Drag and Drop
  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });

  dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('dragover');
  });

  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  // Run Button
  btnRun.addEventListener('click', runInference);

  // Exports
  btnExportCsv.addEventListener('click', exportCsv);
  btnExportJson.addEventListener('click', exportJson);
  if (btnExportCsvTable) btnExportCsvTable.addEventListener('click', exportCsv);
}

function updateBadges() {
  document.getElementById('score-val').textContent = parseFloat(document.getElementById('score-thresh').value).toFixed(2);
  document.getElementById('wbf-val').textContent = parseFloat(document.getElementById('wbf-thresh').value).toFixed(2);
  document.getElementById('tile-val').textContent = document.getElementById('tile-size').value;
}

function handleFileSelected(file) {
  selectedFile = file;
  
  // Show image info box & thumbnail
  const infoBox = document.getElementById('image-info-box');
  const thumb = document.getElementById('thumb-preview');
  const infoName = document.getElementById('info-filename');
  const infoDims = document.getElementById('info-dims');

  infoName.textContent = file.name;
  infoDims.textContent = `${(file.size / (1024 * 1024)).toFixed(2)} MB`;

  const reader = new FileReader();
  reader.onload = (e) => {
    thumb.src = e.target.result;
    infoBox.style.display = 'flex';
    displayMainImage(e.target.result);

    // Read natural dimensions
    const imgObj = new Image();
    imgObj.onload = () => {
      infoDims.textContent = `${imgObj.width} x ${imgObj.height} px`;
    };
    imgObj.src = e.target.result;
  };
  reader.readAsDataURL(file);

  document.getElementById('status-text').textContent = 'Image Loaded';
}

function displayMainImage(src) {
  const placeholder = document.getElementById('placeholder');
  const previewImg = document.getElementById('preview-image');

  placeholder.style.display = 'none';
  previewImg.style.display = 'block';
  previewImg.src = src;
}

async function runInference() {
  if (!selectedFile) {
    alert("Please select or drop an image file first.");
    return;
  }

  const btnRun = document.getElementById('btn-run-inference');
  const statusText = document.getElementById('status-text');

  btnRun.disabled = true;
  btnRun.innerHTML = `<div class="spinner"></div> Running ONNX Inference...`;
  statusText.textContent = 'Processing Tiled ONNX...';

  const scoreThresh = parseFloat(document.getElementById('score-thresh').value);
  const wbfThresh = parseFloat(document.getElementById('wbf-thresh').value);
  const tileSize = parseInt(document.getElementById('tile-size').value);
  const scaleUm = document.getElementById('scale-um').value ? parseFloat(document.getElementById('scale-um').value) : null;

  try {
    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('box_score_thresh', scoreThresh);
    formData.append('wbf_iou_thresh', wbfThresh);
    formData.append('tile_size', tileSize);
    if (scaleUm) formData.append('pixel_to_um', scaleUm);

    const response = await fetch('/api/predict', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      const errData = await response.json();
      throw new Error(errData.error || `Server responded with status ${response.status}`);
    }

    const data = await response.json();
    lastResults = data;

    // Update Viewport with rendered overlay
    if (data.visualization_base64) {
      displayMainImage(`data:image/jpeg;base64,${data.visualization_base64}`);
    }

    updateDashboard(data);
    statusText.textContent = 'Detection Complete';

  } catch (err) {
    console.error("Inference Error:", err);
    alert("Inference Error: " + err.message);
    statusText.textContent = 'Detection Failed';
  } finally {
    btnRun.disabled = false;
    btnRun.innerHTML = `⚡ Run Particle Detection`;
  }
}

function updateDashboard(data) {
  const frags = data.fragments || [];

  // KPI Dashboard Cards
  document.getElementById('kpi-count').textContent = frags.length;

  let unit = "px²";
  let totalArea = 0;
  let meanDia = 0;

  if (frags.length > 0) {
    if (frags[0].area_um2 !== undefined) {
      unit = "μm²";
      totalArea = frags.reduce((sum, f) => sum + f.area_um2, 0);
      meanDia = frags.reduce((sum, f) => sum + f.eq_diameter_um, 0) / frags.length;
      document.getElementById('kpi-scale').textContent = `${data.pixel_to_um.toFixed(4)} μm/px`;
      document.getElementById('scale-badge').textContent = `${data.pixel_to_um.toFixed(3)} μm/px`;
    } else {
      totalArea = frags.reduce((sum, f) => sum + f.area_px, 0);
      meanDia = frags.reduce((sum, f) => sum + f.eq_diameter_px, 0) / frags.length;
      document.getElementById('kpi-scale').textContent = 'Pixels';
      document.getElementById('scale-badge').textContent = 'Pixels';
    }
  }

  document.getElementById('kpi-area').textContent = `${totalArea.toLocaleString(undefined, {maximumFractionDigits: 1})} ${unit}`;
  document.getElementById('kpi-diameter').textContent = `${meanDia.toFixed(1)} ${unit === "μm²" ? "μm" : "px"}`;
  document.getElementById('table-count-label').textContent = `(${frags.length} particles detected)`;

  // Table Body
  const tbody = document.getElementById('table-body');
  tbody.innerHTML = '';

  if (frags.length === 0) {
    tbody.innerHTML = `<tr><td colspan="10" style="text-align: center; color: var(--text-dim); padding: 24px;">No microplastic fragments detected at current thresholds.</td></tr>`;
    return;
  }

  frags.forEach(f => {
    const tr = document.createElement('tr');
    const area = f.area_um2 !== undefined ? f.area_um2 : f.area_px;
    const perim = f.perimeter_um !== undefined ? f.perimeter_um : f.perimeter_px;
    const eqDia = f.eq_diameter_um !== undefined ? f.eq_diameter_um : f.eq_diameter_px;
    const majAx = f.major_axis_um !== undefined ? f.major_axis_um : f.major_axis_px;
    const minAx = f.minor_axis_um !== undefined ? f.minor_axis_um : f.minor_axis_px;

    tr.innerHTML = `
      <td style="color: var(--primary-cyan); font-weight: 600;">#${f.id}</td>
      <td>${(f.score * 100).toFixed(1)}%</td>
      <td>${area.toLocaleString(undefined, {maximumFractionDigits: 1})}</td>
      <td>${perim.toFixed(1)}</td>
      <td>${eqDia.toFixed(1)}</td>
      <td>${majAx.toFixed(1)}</td>
      <td>${minAx.toFixed(1)}</td>
      <td>${f.aspect_ratio.toFixed(2)}</td>
      <td>${f.circularity.toFixed(3)}</td>
      <td>${f.solidity.toFixed(3)}</td>
    `;
    tbody.appendChild(tr);
  });

  // Enable Export Buttons
  document.getElementById('btn-export-csv').disabled = false;
  document.getElementById('btn-export-json').disabled = false;
  const btnExportCsvTable = document.getElementById('btn-export-csv-table');
  if (btnExportCsvTable) btnExportCsvTable.disabled = false;
}

function exportCsv() {
  if (!lastResults || !lastResults.fragments || lastResults.fragments.length === 0) return;
  const frags = lastResults.fragments;

  const headers = Object.keys(frags[0]).join(',');
  const rows = frags.map(f => Object.values(f).map(v => typeof v === 'object' ? `"${JSON.stringify(v)}"` : v).join(','));
  const csvContent = "data:text/csv;charset=utf-8," + [headers, ...rows].join('\n');

  const encodedUri = encodeURI(csvContent);
  const link = document.createElement("a");
  link.setAttribute("href", encodedUri);
  link.setAttribute("download", "mp_fragment_measurements.csv");
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

function exportJson() {
  if (!lastResults) return;
  const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(lastResults, null, 2));
  const link = document.createElement("a");
  link.setAttribute("href", dataStr);
  link.setAttribute("download", "mp_fragment_analysis.json");
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}
