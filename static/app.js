let selectedFiles = []; // Array of File objects
let batchResults = null;
let currentActiveIndex = 0;
let currentRawImageUrls = [];
let showOverlays = true;

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
  const overlapSlider = document.getElementById('overlap-thresh');

  scoreSlider.addEventListener('input', updateBadges);
  wbfSlider.addEventListener('input', updateBadges);
  tileSlider.addEventListener('input', updateBadges);
  if (overlapSlider) overlapSlider.addEventListener('input', updateBadges);
  updateBadges();

  // Click triggers for File Selection
  dropzone.addEventListener('click', () => fileInput.click());
  if (btnBrowse) btnBrowse.addEventListener('click', () => fileInput.click());

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFilesSelected(Array.from(e.target.files));
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
      handleFilesSelected(Array.from(e.dataTransfer.files));
    }
  });

  // Run Button
  btnRun.addEventListener('click', runInference);

  // Exports
  btnExportCsv.addEventListener('click', exportCsv);
  btnExportJson.addEventListener('click', exportJson);
  if (btnExportCsvTable) btnExportCsvTable.addEventListener('click', exportCsv);

  // Sidebar Sliding Toggle
  const btnToggleSidebar = document.getElementById('btn-toggle-sidebar');
  const sidebar = document.querySelector('.sidebar');

  function updateToggleState(isOpen) {
    if (btnToggleSidebar) {
      btnToggleSidebar.classList.toggle('is-open', isOpen);
      btnToggleSidebar.setAttribute('aria-expanded', isOpen);
      btnToggleSidebar.setAttribute('title', isOpen ? 'Hide Controls' : 'Show Controls');
    }
  }

  function toggleSidebar() {
    if (!sidebar) return;
    const isCollapsed = sidebar.classList.toggle('collapsed');
    updateToggleState(!isCollapsed);
  }

  if (btnToggleSidebar) btnToggleSidebar.addEventListener('click', toggleSidebar);

  // Overlay Toggle Button (Eye)
  const btnToggleOverlay = document.getElementById('btn-toggle-overlay');
  if (btnToggleOverlay) {
    btnToggleOverlay.addEventListener('click', () => {
      if (!batchResults || !batchResults.results || batchResults.results.length === 0) return;
      showOverlays = !showOverlays;
      updateOverlayButtonUI();
      displaySingleImageResults(batchResults.results[currentActiveIndex]);
    });
  }

  // Initialize Slidable Table Panel
  initTableResizer();
}

function initTableResizer() {
  const resizer = document.getElementById('table-resizer');
  const tablePanel = document.getElementById('table-panel');
  const btnToggleTable = document.getElementById('btn-toggle-table');
  const workspace = document.querySelector('.workspace');
  if (!resizer || !tablePanel || !workspace) return;

  const DEFAULT_HEIGHT = 240;
  const MIN_HEIGHT = 44; // Keeps table header bar visible
  let lastExpandedHeight = DEFAULT_HEIGHT;

  // Restore saved height from local storage
  try {
    const saved = localStorage.getItem('mp_table_height');
    if (saved) {
      const parsed = parseInt(saved, 10);
      if (!isNaN(parsed) && parsed >= MIN_HEIGHT && parsed <= window.innerHeight * 0.8) {
        tablePanel.style.height = `${parsed}px`;
        if (parsed > 60) lastExpandedHeight = parsed;
      }
    }
  } catch (_) {}

  let isDragging = false;
  let startY = 0;
  let startHeight = 0;

  function updateToggleBtn() {
    if (!btnToggleTable) return;
    const currentH = tablePanel.getBoundingClientRect().height;
    if (currentH <= 60) {
      btnToggleTable.textContent = '▲';
      btnToggleTable.setAttribute('title', 'Expand Table');
    } else {
      btnToggleTable.textContent = '▼';
      btnToggleTable.setAttribute('title', 'Collapse Table (Maximize Viewport)');
    }
  }

  updateToggleBtn();

  resizer.addEventListener('pointerdown', (e) => {
    isDragging = true;
    startY = e.clientY;
    startHeight = tablePanel.getBoundingClientRect().height;
    resizer.classList.add('active');
    document.body.classList.add('is-resizing');
    tablePanel.style.transition = 'none';
    resizer.setPointerCapture(e.pointerId);
  });

  resizer.addEventListener('pointermove', (e) => {
    if (!isDragging) return;
    const dy = startY - e.clientY;
    const workspaceHeight = workspace.clientHeight;
    const maxHeight = Math.max(MIN_HEIGHT, workspaceHeight - 100);
    const newHeight = Math.min(Math.max(startHeight + dy, MIN_HEIGHT), maxHeight);
    tablePanel.style.height = `${Math.round(newHeight)}px`;
    updateToggleBtn();
  });

  function endDrag(e) {
    if (!isDragging) return;
    isDragging = false;
    resizer.classList.remove('active');
    document.body.classList.remove('is-resizing');
    tablePanel.style.transition = '';
    const finalH = Math.round(tablePanel.getBoundingClientRect().height);
    if (finalH > 60) {
      lastExpandedHeight = finalH;
    }
    try {
      localStorage.setItem('mp_table_height', finalH);
      resizer.releasePointerCapture(e.pointerId);
    } catch (_) {}
    updateToggleBtn();
  }

  resizer.addEventListener('pointerup', endDrag);
  resizer.addEventListener('pointercancel', endDrag);

  function toggleTableHeight() {
    tablePanel.style.transition = 'height 0.28s cubic-bezier(0.4, 0, 0.2, 1)';
    const currentH = tablePanel.getBoundingClientRect().height;
    if (currentH <= 60) {
      // Restore / Expand
      const restoreH = lastExpandedHeight > 60 ? lastExpandedHeight : DEFAULT_HEIGHT;
      tablePanel.style.height = `${restoreH}px`;
      try { localStorage.setItem('mp_table_height', restoreH); } catch (_) {}
    } else {
      // Collapse
      lastExpandedHeight = currentH;
      tablePanel.style.height = `${MIN_HEIGHT}px`;
      try { localStorage.setItem('mp_table_height', MIN_HEIGHT); } catch (_) {}
    }
    setTimeout(() => {
      tablePanel.style.transition = '';
      updateToggleBtn();
    }, 300);
  }

  resizer.addEventListener('dblclick', toggleTableHeight);
  if (btnToggleTable) btnToggleTable.addEventListener('click', toggleTableHeight);
}

function updateBadges() {
  document.getElementById('score-val').textContent = parseFloat(document.getElementById('score-thresh').value).toFixed(2);
  document.getElementById('wbf-val').textContent = parseFloat(document.getElementById('wbf-thresh').value).toFixed(2);
  document.getElementById('tile-val').textContent = document.getElementById('tile-size').value;
  const overlapEl = document.getElementById('overlap-thresh');
  if (overlapEl) {
    document.getElementById('overlap-val').textContent = `${Math.round(parseFloat(overlapEl.value) * 100)}%`;
  }
}

function updateOverlayButtonUI() {
  const btn = document.getElementById('btn-toggle-overlay');
  if (!btn) return;
  const eyeOpen = btn.querySelector('.eye-open');
  const eyeClosed = btn.querySelector('.eye-closed');
  if (showOverlays) {
    btn.classList.add('is-active');
    btn.classList.remove('is-hidden-mode');
    btn.title = "Hide Detection Overlays (Show Raw Image)";
    if (eyeOpen) eyeOpen.style.display = 'block';
    if (eyeClosed) eyeClosed.style.display = 'none';
  } else {
    btn.classList.remove('is-active');
    btn.classList.add('is-hidden-mode');
    btn.title = "Show Detection Overlays";
    if (eyeOpen) eyeOpen.style.display = 'none';
    if (eyeClosed) eyeClosed.style.display = 'block';
  }
}

function updateOverlayToggleState(enabled) {
  const btn = document.getElementById('btn-toggle-overlay');
  if (!btn) return;
  btn.disabled = !enabled;
  if (enabled) {
    updateOverlayButtonUI();
  } else {
    btn.classList.remove('is-active', 'is-hidden-mode');
    btn.title = "Run inference first to toggle overlays";
    const eyeOpen = btn.querySelector('.eye-open');
    const eyeClosed = btn.querySelector('.eye-closed');
    if (eyeOpen) eyeOpen.style.display = 'block';
    if (eyeClosed) eyeClosed.style.display = 'none';
  }
}

function handleFilesSelected(files) {
  selectedFiles = files;
  currentRawImageUrls.forEach(url => URL.revokeObjectURL(url));
  currentRawImageUrls = files.map(f => URL.createObjectURL(f));
  batchResults = null;
  updateOverlayToggleState(false);
  
  const infoBox = document.getElementById('image-info-box');
  const thumb = document.getElementById('thumb-preview');
  const infoName = document.getElementById('info-filename');
  const infoDims = document.getElementById('info-dims');

  if (files.length === 1) {
    infoName.textContent = files[0].name;
    infoDims.textContent = `${(files[0].size / (1024 * 1024)).toFixed(2)} MB`;

    const reader = new FileReader();
    reader.onload = (e) => {
      thumb.src = e.target.result;
      infoBox.style.display = 'flex';
      displayMainImage(e.target.result);
    };
    reader.readAsDataURL(files[0]);
  } else {
    infoName.textContent = `${files.length} Images Selected`;
    infoDims.textContent = `Batch size: ${files.length} scans`;
    infoBox.style.display = 'flex';

    const reader = new FileReader();
    reader.onload = (e) => {
      thumb.src = e.target.result;
      displayMainImage(e.target.result);
    };
    reader.readAsDataURL(files[0]);
  }

  document.getElementById('status-text').textContent = `${files.length} Image(s) Loaded`;
}

function displayMainImage(src) {
  const placeholder = document.getElementById('placeholder');
  const previewImg = document.getElementById('preview-image');

  placeholder.style.display = 'none';
  previewImg.style.display = 'block';
  previewImg.src = src;
}

async function runInference() {
  if (!selectedFiles || selectedFiles.length === 0) {
    alert("Please select or drop 1 or more image files first.");
    return;
  }

  const btnRun = document.getElementById('btn-run-inference');
  const statusText = document.getElementById('status-text');

  btnRun.disabled = true;
  btnRun.innerHTML = `<div class="spinner"></div> Running Inference...`;
  statusText.textContent = `Processing ${selectedFiles.length} Scan(s)...`;

  const scoreThresh = parseFloat(document.getElementById('score-thresh').value);
  const wbfThresh = parseFloat(document.getElementById('wbf-thresh').value);
  const tileSize = parseInt(document.getElementById('tile-size').value);
  const overlapThresh = document.getElementById('overlap-thresh') ? parseFloat(document.getElementById('overlap-thresh').value) : 0.3;
  const scaleUm = document.getElementById('scale-um').value ? parseFloat(document.getElementById('scale-um').value) : null;

  try {
    const formData = new FormData();
    selectedFiles.forEach(file => {
      formData.append('files', file);
    });
    formData.append('box_score_thresh', scoreThresh);
    formData.append('wbf_iou_thresh', wbfThresh);
    formData.append('tile_size', tileSize);
    formData.append('overlap_pct', overlapThresh);
    if (scaleUm) formData.append('pixel_to_um', scaleUm);

    const response = await fetch('/api/predict_batch', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      const errData = await response.json();
      throw new Error(errData.error || `Server responded with status ${response.status}`);
    }

    const data = await response.json();
    batchResults = data;
    currentActiveIndex = 0;

    updateUIWithBatchData(data);
    statusText.textContent = 'Batch Detection Complete';

  } catch (err) {
    console.error("Inference Error:", err);
    alert("Inference Error: " + err.message);
    statusText.textContent = 'Detection Failed';
  } finally {
    btnRun.disabled = false;
    btnRun.innerHTML = `Run Inference`;
  }
}

function updateUIWithBatchData(data) {
  const summary = data.batch_summary || {};
  const results = data.results || [];

  if (results.length === 0) return;

  // Build Batch Carousel Selector if multiple images
  const carousel = document.getElementById('batch-carousel');
  const carouselItems = document.getElementById('carousel-items');
  carouselItems.innerHTML = '';

  if (results.length > 1) {
    carousel.style.display = 'flex';
    results.forEach((res, idx) => {
      const item = document.createElement('div');
      item.className = `carousel-item ${idx === currentActiveIndex ? 'active' : ''}`;
      item.innerHTML = `<span>📄</span> ${res.image_name} (${res.fragments.length})`;
      item.addEventListener('click', () => {
        currentActiveIndex = idx;
        document.querySelectorAll('.carousel-item').forEach((el, i) => {
          el.classList.toggle('active', i === idx);
        });
        displaySingleImageResults(results[idx]);
      });
      carouselItems.appendChild(item);
    });
  } else {
    carousel.style.display = 'none';
  }

  // Display initial active image
  showOverlays = true;
  updateOverlayToggleState(true);
  displaySingleImageResults(results[currentActiveIndex]);

  // Aggregate Batch KPI Cards
  document.getElementById('kpi-count').textContent = summary.total_fragments || 0;
  document.getElementById('kpi-area').textContent = `${(summary.total_area || 0).toLocaleString(undefined, {maximumFractionDigits: 1})} μm²`;

  // Compute dominant color overall
  const colorDist = summary.color_distribution || {};
  let topColor = "-";
  let topCount = 0;
  for (const [col, count] of Object.entries(colorDist)) {
    if (count > topCount) {
      topCount = count;
      topColor = col;
    }
  }
  document.getElementById('kpi-color').textContent = topColor;
}

function displaySingleImageResults(resData) {
  if (showOverlays && resData.visualization_base64) {
    displayMainImage(`data:image/jpeg;base64,${resData.visualization_base64}`);
  } else if (currentRawImageUrls[currentActiveIndex]) {
    displayMainImage(currentRawImageUrls[currentActiveIndex]);
  } else if (resData.visualization_base64) {
    displayMainImage(`data:image/jpeg;base64,${resData.visualization_base64}`);
  }

  const frags = resData.fragments || [];
  if (resData.scalebar_info && resData.scalebar_info.detected) {
    const sb = resData.scalebar_info;
    const valText = sb.scale_um >= 1000 && sb.unit === 'mm' ? `${sb.scale_um / 1000} mm` : `${sb.scale_um} μm`;
    document.getElementById('kpi-scale').textContent = `${valText} (${Math.round(sb.pixel_width)}px) → ${resData.pixel_to_um.toFixed(3)} μm/px`;
  } else if (resData.pixel_to_um) {
    document.getElementById('kpi-scale').textContent = `${resData.pixel_to_um.toFixed(3)} μm/px`;
  } else {
    document.getElementById('kpi-scale').textContent = 'Auto';
  }
  document.getElementById('table-count-label').textContent = `(${frags.length} particles in ${resData.image_name})`;

  // Populate Table Body
  const tbody = document.getElementById('table-body');
  tbody.innerHTML = '';

  if (frags.length === 0) {
    tbody.innerHTML = `<tr><td colspan="12" style="text-align: center; color: var(--text-dim); padding: 24px;">No fragments detected in this scan.</td></tr>`;
    return;
  }

  frags.forEach(f => {
    const tr = document.createElement('tr');
    const area = f.area_um2 !== undefined ? f.area_um2 : f.area_px;
    const perim = f.perimeter_um !== undefined ? f.perimeter_um : f.perimeter_px;
    const eqDia = f.eq_diameter_um !== undefined ? f.eq_diameter_um : f.eq_diameter_px;
    const majAx = f.major_axis_um !== undefined ? f.major_axis_um : f.major_axis_px;
    const minAx = f.minor_axis_um !== undefined ? f.minor_axis_um : f.minor_axis_px;
    const hex = f.hex_code || "#808080";
    const colorName = f.color_name || "Unknown";

    tr.innerHTML = `
      <td style="color: var(--primary-cyan); font-weight: 600;">${f.id}</td>
      <td style="font-size: 0.75rem; color: var(--text-muted);">${f.image_name}</td>
      <td>
        <span class="color-badge">
          <span class="color-dot" style="background-color: ${hex};"></span>
          ${colorName}
        </span>
      </td>
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

  document.getElementById('btn-export-csv').disabled = false;
  document.getElementById('btn-export-json').disabled = false;
  const btnExportCsvTable = document.getElementById('btn-export-csv-table');
  if (btnExportCsvTable) btnExportCsvTable.disabled = false;
}

function exportCsv() {
  if (!batchResults || !batchResults.results || batchResults.results.length === 0) return;

  // Flatten all fragments across all processed images in the batch
  const allFrags = [];
  batchResults.results.forEach(res => {
    res.fragments.forEach(f => {
      allFrags.push(f);
    });
  });

  if (allFrags.length === 0) return;

  const headers = Object.keys(allFrags[0]).join(',');
  const rows = allFrags.map(f => Object.values(f).map(v => typeof v === 'object' ? `"${JSON.stringify(v)}"` : v).join(','));
  const csvContent = "data:text/csv;charset=utf-8," + [headers, ...rows].join('\n');

  const encodedUri = encodeURI(csvContent);
  const link = document.createElement("a");
  link.setAttribute("href", encodedUri);
  link.setAttribute("download", "mp_fragment_batch_measurements.csv");
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

function exportJson() {
  if (!batchResults) return;
  const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(batchResults, null, 2));
  const link = document.createElement("a");
  link.setAttribute("href", dataStr);
  link.setAttribute("download", "mp_fragment_batch_analysis.json");
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}
