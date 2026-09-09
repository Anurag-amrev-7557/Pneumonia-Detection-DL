/**
 * PULMO·AI Enterprise Clinical Decision Support Engine
 * Bilateral Side-by-Side PACS Viewport, Collapsible Worklist, and Clinician Actions
 */

(function () {
  'use strict';

  // State Store
  const state = {
    confidenceThreshold: 0.50,
    cropMargins: false,
    currentImage: null,          // HTMLImageElement
    currentHeatmap: null,        // HTMLImageElement
    currentSampleId: null,
    currentFile: null,
    currentPrediction: null,
    
    // Viewport Mode: 'side-by-side' (default) vs 'overlay'
    viewMode: 'side-by-side',
    isWorklistCollapsed: false,
    clinicianDetermination: 'Pending Review',
    
    // Viewport transforms
    scale: 1.0,
    baseFitScale: 1.0,
    panX: 0,
    panY: 0,
    isDragging: false,
    lastMouseX: 0,
    lastMouseY: 0,
    isInverted: false,
    heatmapAlpha: 0.45,
  };

  // DOM Cache
  const dom = {
    canvas: document.getElementById('pacsCanvas'),
    canvasContainer: document.getElementById('canvasContainer'),
    viewportPlaceholder: document.getElementById('viewportPlaceholder'),
    loadingOverlay: document.getElementById('loadingOverlay'),
    
    // Viewport Controls
    btnViewSideBySide: document.getElementById('btnViewSideBySide'),
    btnViewOverlay: document.getElementById('btnViewOverlay'),
    alphaSliderContainer: document.getElementById('alphaSliderContainer'),
    heatmapAlphaSlider: document.getElementById('heatmapAlphaSlider'),
    alphaValueDisplay: document.getElementById('alphaValueDisplay'),
    btnZoomIn: document.getElementById('btnZoomIn'),
    btnZoomOut: document.getElementById('btnZoomOut'),
    btnResetView: document.getElementById('btnResetView'),
    btnInvert: document.getElementById('btnInvert'),
    toggleWorklistBtn: document.getElementById('toggleWorklistBtn'),
    toggleWorklistIcon: document.getElementById('toggleWorklistIcon'),
    bottomWorklistSection: document.getElementById('bottomWorklistSection'),
    
    // Mode Buttons
    modeBalanced: document.getElementById('modeBalanced'),
    modeScreening: document.getElementById('modeScreening'),
    cropMarginsToggle: document.getElementById('cropMarginsToggle'),
    badgeThreshold: document.getElementById('badgeThreshold'),
    
    // Viewport Overlays
    overlayDimensions: document.getElementById('overlayDimensions'),
    overlayZoom: document.getElementById('overlayZoom'),
    overlayLatency: document.getElementById('overlayLatency'),
    
    // Diagnostic Telemetry
    triageHero: document.getElementById('triageHero'),
    diagnosisLabel: document.getElementById('diagnosisLabel'),
    diagnosisSubtext: document.getElementById('diagnosisSubtext'),
    probPneumonia: document.getElementById('probPneumonia'),
    riskMeterBar: document.getElementById('riskMeterBar'),
    borderlineBanner: document.getElementById('borderlineBanner'),
    
    // Clinician Sign-off
    clinicianActionStatus: document.getElementById('clinicianActionStatus'),
    btnConcurAI: document.getElementById('btnConcurAI'),
    btnFlagReview: document.getElementById('btnFlagReview'),
    
    // Consensus
    consensusBadge: document.getElementById('consensusBadge'),
    resScore: document.getElementById('resScore'),
    resBar: document.getElementById('resBar'),
    denseScore: document.getElementById('denseScore'),
    denseBar: document.getElementById('denseBar'),
    
    // Localization
    attentionFocus: document.getElementById('attentionFocus'),
    orientationStatus: document.getElementById('orientationStatus'),
    
    // Sample List & Upload
    samplesList: document.getElementById('samplesList'),
    miniDropzone: document.getElementById('miniDropzone'),
    triggerUploadBtn: document.getElementById('triggerUploadBtn'),
    fileInput: document.getElementById('fileInput'),
    
    // Report Modal
    exportReportBtn: document.getElementById('exportReportBtn'),
    reportModal: document.getElementById('reportModal'),
    closeModalBtn: document.getElementById('closeModalBtn'),
    printBtn: document.getElementById('printBtn'),
    reportTime: document.getElementById('reportTime'),
    reportPatientId: document.getElementById('reportPatientId'),
    reportCutoff: document.getElementById('reportCutoff'),
    reportLatency: document.getElementById('reportLatency'),
    reportImpression: document.getElementById('reportImpression'),
    reportClinicianStatus: document.getElementById('reportClinicianStatus'),
    reportResNetProb: document.getElementById('reportResNetProb'),
    reportDenseNetProb: document.getElementById('reportDenseNetProb'),
    reportEnsembleProb: document.getElementById('reportEnsembleProb'),
  };

  const ctx = dom.canvas.getContext('2d');

  // ================= INITIALIZATION =================
  function init() {
    setupCanvas();
    setupEventListeners();
    fetchSamples();
    refreshIcons();
    
    // Auto-resize and re-center on any container size change (window resize, worklist collapse/expand)
    if (window.ResizeObserver) {
      const ro = new ResizeObserver(() => {
        setupCanvas();
        fitImageToViewport();
        renderCanvas();
      });
      ro.observe(dom.canvasContainer);
    } else {
      window.addEventListener('resize', () => {
        setupCanvas();
        fitImageToViewport();
        renderCanvas();
      });
    }
  }

  function refreshIcons() {
    if (window.lucide) {
      window.lucide.createIcons();
    }
  }

  // ================= CANVAS SETUP & RENDERING =================
  function setupCanvas() {
    const width = dom.canvasContainer.clientWidth;
    const height = dom.canvasContainer.clientHeight;
    if (width === 0 || height === 0) return;
    
    const dpr = window.devicePixelRatio || 1;
    dom.canvas.width = Math.round(width * dpr);
    dom.canvas.height = Math.round(height * dpr);
    dom.canvas.style.width = `${width}px`;
    dom.canvas.style.height = `${height}px`;
  }

  function fitImageToViewport() {
    if (!state.currentImage) return;
    const width = dom.canvasContainer.clientWidth;
    const height = dom.canvasContainer.clientHeight;
    if (width === 0 || height === 0) return;

    // Margins: Top reserves 56px for HUD labels, Bottom reserves 36px, Sides reserve 36px
    const paddingX = 40;
    const paddingTop = 60;
    const paddingBottom = 36;
    
    const availWidth = Math.max(100, width - paddingX * 2);
    const availHeight = Math.max(100, height - paddingTop - paddingBottom);
    
    const imgW = state.currentImage.naturalWidth;
    const imgH = state.currentImage.naturalHeight;

    if (state.viewMode === 'side-by-side') {
      // In side-by-side, each image occupies half the width minus central gap (24px)
      const halfWidth = (availWidth - 24) / 2;
      const scaleX = halfWidth / imgW;
      const scaleY = availHeight / imgH;
      state.baseFitScale = Math.min(scaleX, scaleY, 1.25);
    } else {
      const scaleX = availWidth / imgW;
      const scaleY = availHeight / imgH;
      state.baseFitScale = Math.min(scaleX, scaleY, 1.35);
    }

    state.scale = state.baseFitScale;
    state.panX = 0;
    
    // Exact vertical centering in the available space between paddingTop and paddingBottom
    const availableCenterY = paddingTop + availHeight / 2;
    state.panY = availableCenterY - (height / 2);
  }

  function renderCanvas() {
    const width = dom.canvasContainer.clientWidth;
    const height = dom.canvasContainer.clientHeight;
    if (width === 0 || height === 0) return;
    
    const dpr = window.devicePixelRatio || 1;

    // 1. Reset transform to absolute identity
    ctx.setTransform(1, 0, 0, 1, 0, 0);

    // 2. Clear 100% of physical backing store pixels
    ctx.clearRect(0, 0, dom.canvas.width, dom.canvas.height);

    // 3. Fill clean light medical lightbox background
    ctx.fillStyle = '#F1F5F9';
    ctx.fillRect(0, 0, dom.canvas.width, dom.canvas.height);

    // 4. Apply DPR scaling for crisp Retina rendering
    ctx.scale(dpr, dpr);

    if (!state.currentImage) return;

    const imgW = state.currentImage.naturalWidth;
    const imgH = state.currentImage.naturalHeight;

    if (state.viewMode === 'side-by-side') {
      renderSideBySide(width, height, imgW, imgH);
    } else {
      renderSingleOverlay(width, height, imgW, imgH);
    }

    // Update Zoom percentage in HUD
    const zoomPercent = Math.round((state.scale / (state.baseFitScale || 1)) * 100);
    dom.overlayZoom.textContent = `${zoomPercent}%`;
  }

  function renderSingleOverlay(width, height, imgW, imgH) {
    ctx.save();
    ctx.translate(width / 2 + state.panX, height / 2 + state.panY);
    ctx.scale(state.scale, state.scale);

    const x = -imgW / 2;
    const y = -imgH / 2;

    // Soft drop shadow behind radiograph
    ctx.save();
    ctx.shadowColor = 'rgba(15, 23, 42, 0.22)';
    ctx.shadowBlur = 24 / state.scale;
    ctx.shadowOffsetX = 0;
    ctx.shadowOffsetY = 6 / state.scale;
    ctx.fillStyle = '#000000';
    ctx.fillRect(x, y, imgW, imgH);
    ctx.restore();

    // Radiograph base
    if (state.isInverted) {
      ctx.filter = 'invert(100%) contrast(115%)';
    } else {
      ctx.filter = 'contrast(102%)';
    }
    ctx.drawImage(state.currentImage, x, y, imgW, imgH);
    ctx.filter = 'none';

    // Heatmap overlay
    if (state.currentHeatmap && state.heatmapAlpha > 0) {
      ctx.globalAlpha = state.heatmapAlpha;
      ctx.drawImage(state.currentHeatmap, x, y, imgW, imgH);
      ctx.globalAlpha = 1.0;
    }

    // Crisp high-contrast border
    ctx.strokeStyle = '#94A3B8';
    ctx.lineWidth = 1.5 / state.scale;
    ctx.strokeRect(x, y, imgW, imgH);

    ctx.restore();
  }

  function renderSideBySide(width, height, imgW, imgH) {
    const gap = 24 * state.scale;
    const halfGap = gap / 2;

    ctx.save();
    ctx.translate(width / 2 + state.panX, height / 2 + state.panY);
    ctx.scale(state.scale, state.scale);

    const leftX = -imgW - (halfGap / state.scale);
    const rightX = (halfGap / state.scale);
    const topY = -imgH / 2;

    // Drop shadow behind both images
    ctx.save();
    ctx.shadowColor = 'rgba(15, 23, 42, 0.22)';
    ctx.shadowBlur = 20 / state.scale;
    ctx.shadowOffsetX = 0;
    ctx.shadowOffsetY = 5 / state.scale;
    ctx.fillStyle = '#000000';
    ctx.fillRect(leftX, topY, imgW, imgH);
    ctx.fillRect(rightX, topY, imgW, imgH);
    ctx.restore();

    // Left Image (Pure Anatomical Ground Truth)
    if (state.isInverted) {
      ctx.filter = 'invert(100%) contrast(115%)';
    } else {
      ctx.filter = 'contrast(102%)';
    }
    ctx.drawImage(state.currentImage, leftX, topY, imgW, imgH);
    ctx.filter = 'none';

    // Right Image (Radiograph + Calibrated Grad-CAM Overlay)
    if (state.isInverted) {
      ctx.filter = 'invert(100%) contrast(115%)';
    } else {
      ctx.filter = 'contrast(102%)';
    }
    ctx.drawImage(state.currentImage, rightX, topY, imgW, imgH);
    ctx.filter = 'none';

    if (state.currentHeatmap) {
      ctx.globalAlpha = 0.55;
      ctx.drawImage(state.currentHeatmap, rightX, topY, imgW, imgH);
      ctx.globalAlpha = 1.0;
    }

    // High-contrast borders
    ctx.strokeStyle = '#94A3B8';
    ctx.lineWidth = 1.5 / state.scale;
    ctx.strokeRect(leftX, topY, imgW, imgH);
    ctx.strokeRect(rightX, topY, imgW, imgH);

    ctx.restore();

    // High-Contrast Light Viewport HUD Labels
    ctx.save();
    ctx.font = 'bold 11px Inter, sans-serif';
    
    // Left Label Box
    ctx.fillStyle = '#FFFFFF';
    ctx.fillRect(16, 16, 224, 28);
    ctx.strokeStyle = '#94A3B8';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(16, 16, 224, 28);
    ctx.fillStyle = '#0F172A';
    ctx.fillText('ANATOMICAL RADIOGRAPH (AP/PA)', 24, 34);

    // Right Label Box
    const rightLabelX = Math.max(width / 2 + 10, width - 230);
    ctx.fillStyle = '#FFFFFF';
    ctx.fillRect(rightLabelX, 16, 214, 28);
    ctx.strokeStyle = '#94A3B8';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(rightLabelX, 16, 214, 28);
    ctx.fillStyle = '#1E40AF';
    ctx.fillText('GRAD-CAM ATTENTION OVERLAY', rightLabelX + 12, 34);

    // Subtle Center Divider Line
    ctx.strokeStyle = '#CBD5E1';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(width / 2, 0);
    ctx.lineTo(width / 2, height);
    ctx.stroke();

    ctx.restore();
  }

  // ================= EVENT LISTENERS =================
  function setupEventListeners() {
    // Mode Switcher: Side-by-Side vs Overlay
    dom.btnViewSideBySide.addEventListener('click', () => {
      state.viewMode = 'side-by-side';
      dom.btnViewSideBySide.className = 'px-2.5 py-1 rounded text-xs font-extrabold transition-all bg-white text-blue-900 shadow-xs border border-slate-400 flex items-center gap-1.5';
      dom.btnViewOverlay.className = 'px-2.5 py-1 rounded text-xs font-bold transition-all text-slate-700 hover:text-slate-950 flex items-center gap-1.5';
      fitImageToViewport();
      renderCanvas();
    });

    dom.btnViewOverlay.addEventListener('click', () => {
      state.viewMode = 'overlay';
      dom.btnViewOverlay.className = 'px-2.5 py-1 rounded text-xs font-extrabold transition-all bg-white text-blue-900 shadow-xs border border-slate-400 flex items-center gap-1.5';
      dom.btnViewSideBySide.className = 'px-2.5 py-1 rounded text-xs font-bold transition-all text-slate-700 hover:text-slate-950 flex items-center gap-1.5';
      fitImageToViewport();
      renderCanvas();
    });

    // Collapsible Bottom Worklist
    dom.toggleWorklistBtn.addEventListener('click', () => {
      state.isWorklistCollapsed = !state.isWorklistCollapsed;
      if (state.isWorklistCollapsed) {
        dom.bottomWorklistSection.classList.add('hidden');
        dom.toggleWorklistBtn.setAttribute('title', 'Expand Patient Worklist');
        dom.toggleWorklistIcon.setAttribute('data-lucide', 'chevron-up');
      } else {
        dom.bottomWorklistSection.classList.remove('hidden');
        dom.toggleWorklistBtn.setAttribute('title', 'Maximize Viewport (Collapse Worklist)');
        dom.toggleWorklistIcon.setAttribute('data-lucide', 'chevron-down');
      }
      refreshIcons();
      setTimeout(() => {
        setupCanvas();
        fitImageToViewport();
        renderCanvas();
      }, 50);
    });

    // Clinician Decision Actions & Persistent Audit Logging
    async function logFeedbackToBackend(action) {
      if (!state.currentPrediction) return;
      const caseId = state.currentSampleId || (state.currentFile ? state.currentFile.name : 'uploaded-case');
      try {
        await fetch('/api/feedback', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            case_id: caseId,
            clinician_action: action,
            ai_prediction: state.currentPrediction.label,
            ai_confidence: state.currentPrediction.confidence,
            threshold_used: state.currentPrediction.threshold_used,
            notes: action === 'concur' ? 'Attending concurred with AI triage' : 'Flagged for secondary senior radiologist audit'
          })
        });
      } catch (err) {
        console.warn('Could not log clinician feedback to audit log:', err);
      }
    }

    dom.btnConcurAI.addEventListener('click', () => {
      state.clinicianDetermination = 'Concurred with AI Determination';
      dom.clinicianActionStatus.textContent = 'Logged: Concurred';
      dom.clinicianActionStatus.className = 'text-[11px] font-black px-2.5 py-0.5 rounded-full bg-emerald-100 text-emerald-950 border border-emerald-600';
      dom.reportClinicianStatus.textContent = 'Clinician Concurred with Automated AI Findings';
      dom.reportClinicianStatus.className = 'text-emerald-950 bg-emerald-100 px-3 py-1 rounded-full border border-emerald-600 font-black';
      logFeedbackToBackend('concur');
    });

    dom.btnFlagReview.addEventListener('click', () => {
      state.clinicianDetermination = 'Flagged for Secondary Radiologist Review';
      dom.clinicianActionStatus.textContent = 'Logged: Flagged';
      dom.clinicianActionStatus.className = 'text-[11px] font-black px-2.5 py-0.5 rounded-full bg-amber-100 text-amber-950 border border-amber-600';
      dom.reportClinicianStatus.textContent = 'Flagged for Secondary Clinical & Radiologic Review';
      dom.reportClinicianStatus.className = 'text-amber-950 bg-amber-100 px-3 py-1 rounded-full border border-amber-600 font-black';
      logFeedbackToBackend('flag_for_review');
    });

    // Zoom In / Out / Reset
    dom.btnZoomIn.addEventListener('click', () => {
      state.scale = Math.min(state.scale * 1.25, 12.0);
      renderCanvas();
    });

    dom.btnZoomOut.addEventListener('click', () => {
      state.scale = Math.max(state.scale / 1.25, 0.15);
      renderCanvas();
    });

    dom.btnResetView.addEventListener('click', () => {
      fitImageToViewport();
      renderCanvas();
    });

    // Invert Grayscale Toggle
    dom.btnInvert.addEventListener('click', () => {
      state.isInverted = !state.isInverted;
      if (state.isInverted) {
        dom.btnInvert.className = 'flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-950 text-white border border-slate-950 text-xs font-bold transition shadow-xs';
      } else {
        dom.btnInvert.className = 'flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-white border border-slate-400 text-slate-900 hover:bg-slate-100 text-xs font-bold transition shadow-xs';
      }
      renderCanvas();
    });

    // Alpha Slider
    dom.heatmapAlphaSlider.addEventListener('input', (e) => {
      state.heatmapAlpha = parseInt(e.target.value, 10) / 100;
      dom.alphaValueDisplay.textContent = `${e.target.value}%`;
      renderCanvas();
    });

    // Canvas Pan & Zoom
    dom.canvasContainer.addEventListener('wheel', (e) => {
      e.preventDefault();
      const zoomFactor = e.deltaY < 0 ? 1.12 : 0.89;
      state.scale = Math.max(0.15, Math.min(12.0, state.scale * zoomFactor));
      renderCanvas();
    }, { passive: false });

    dom.canvasContainer.addEventListener('mousedown', (e) => {
      if (!state.currentImage) return;
      state.isDragging = true;
      state.lastMouseX = e.clientX;
      state.lastMouseY = e.clientY;
    });

    window.addEventListener('mousemove', (e) => {
      if (!state.isDragging) return;
      const dx = e.clientX - state.lastMouseX;
      const dy = e.clientY - state.lastMouseY;
      state.panX += dx;
      state.panY += dy;
      state.lastMouseX = e.clientX;
      state.lastMouseY = e.clientY;
      renderCanvas();
    });

    window.addEventListener('mouseup', () => {
      state.isDragging = false;
    });

    // Sensitivity mode toggle
    dom.modeBalanced.addEventListener('click', () => {
      setSensitivityMode(0.50);
    });

    dom.modeScreening.addEventListener('click', () => {
      setSensitivityMode(0.35);
    });

    // Margin crop toggle
    dom.cropMarginsToggle.addEventListener('change', (e) => {
      state.cropMargins = e.target.checked;
      reRunCurrentAnalysis();
    });

    // File Upload Handlers
    dom.triggerUploadBtn.addEventListener('click', () => dom.fileInput.click());
    dom.miniDropzone.addEventListener('click', () => dom.fileInput.click());

    dom.fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) {
        handleFileUpload(e.target.files[0]);
      }
    });

    // Drag & Drop
    const preventDefaults = (e) => { e.preventDefault(); e.stopPropagation(); };
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(evt => {
      dom.canvasContainer.addEventListener(evt, preventDefaults, false);
      dom.miniDropzone.addEventListener(evt, preventDefaults, false);
    });

    dom.canvasContainer.addEventListener('drop', (e) => {
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleFileUpload(e.dataTransfer.files[0]);
      }
    });

    dom.miniDropzone.addEventListener('drop', (e) => {
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleFileUpload(e.dataTransfer.files[0]);
      }
    });

    // Clinical Report Modal
    dom.exportReportBtn.addEventListener('click', openReportModal);
    dom.closeModalBtn.addEventListener('click', closeReportModal);
    dom.reportModal.addEventListener('click', (e) => {
      if (e.target === dom.reportModal) closeReportModal();
    });
    dom.printBtn.addEventListener('click', () => window.print());
  }

  function setSensitivityMode(threshold) {
    state.confidenceThreshold = threshold;
    const isBalanced = threshold === 0.50;
    
    if (isBalanced) {
      dom.modeBalanced.className = 'px-3.5 py-1.5 rounded-lg text-xs font-bold transition-all bg-white text-blue-900 shadow-xs border border-slate-400';
      dom.modeScreening.className = 'px-3.5 py-1.5 rounded-lg text-xs font-bold transition-all text-slate-700 hover:text-slate-950 hover:bg-slate-200';
    } else {
      dom.modeBalanced.className = 'px-3.5 py-1.5 rounded-lg text-xs font-bold transition-all text-slate-700 hover:text-slate-950 hover:bg-slate-200';
      dom.modeScreening.className = 'px-3.5 py-1.5 rounded-lg text-xs font-bold transition-all bg-amber-200 text-amber-950 shadow-xs border border-amber-600';
    }

    dom.badgeThreshold.textContent = `CUTOFF: ${Math.round(threshold * 100)}%`;
    reRunCurrentAnalysis();
  }

  function reRunCurrentAnalysis() {
    if (state.currentFile) {
      handleFileUpload(state.currentFile);
    } else if (state.currentSampleId) {
      loadSample(state.currentSampleId);
    }
  }

  // ================= API CALLS & DATA HANDLING =================
  async function fetchSamples() {
    try {
      const res = await fetch('/api/samples');
      const data = await res.json();
      renderSampleCards(data.samples || []);
      
      // Auto-load first sample
      if (data.samples && data.samples.length > 0) {
        loadSample(data.samples[0].id);
      }
    } catch (err) {
      console.error('Failed to fetch sample radiographs:', err);
    }
  }

  function renderSampleCards(samples) {
    dom.samplesList.innerHTML = '';
    // Display top 4 samples to maintain exact 4:1 symmetrical grid with dropzone
    samples.slice(0, 4).forEach(sample => {
      const card = document.createElement('div');
      card.className = `sample-card p-3 rounded-xl border border-slate-300 bg-white hover:bg-slate-50 cursor-pointer flex flex-col justify-between select-none shadow-2xs`;
      card.dataset.sampleId = sample.id;
      
      const badgeStyle = sample.badge_color === 'rose'
        ? 'bg-red-100 text-red-950 border border-red-600 font-extrabold'
        : sample.badge_color === 'amber'
        ? 'bg-amber-100 text-amber-950 border border-amber-600 font-extrabold'
        : sample.badge_color === 'purple'
        ? 'bg-purple-100 text-purple-950 border border-purple-600 font-extrabold'
        : 'bg-emerald-100 text-emerald-950 border border-emerald-600 font-extrabold';

      card.innerHTML = `
        <div>
          <div class="flex items-center justify-between mb-1.5">
            <span class="text-xs font-black text-slate-950">${sample.title}</span>
            <span class="text-[10px] uppercase px-2 py-0.5 rounded ${badgeStyle}">${sample.type}</span>
          </div>
          <p class="text-xs font-semibold text-slate-700 line-clamp-2 leading-relaxed">${sample.description}</p>
        </div>
        <div class="mt-2 text-xs font-bold text-blue-800 flex items-center gap-1">
          <span>Evaluate case</span>
          <i data-lucide="chevron-right" class="w-3.5 h-3.5 text-blue-800"></i>
        </div>
      `;

      card.addEventListener('click', () => {
        document.querySelectorAll('.sample-card').forEach(c => {
          c.classList.remove('active', 'border-blue-700', 'bg-blue-50', 'ring-2', 'ring-blue-700');
        });
        card.classList.add('active', 'border-blue-700', 'bg-blue-50', 'ring-2', 'ring-blue-700');
        loadSample(sample.id);
      });

      dom.samplesList.appendChild(card);
    });
    refreshIcons();
  }

  async function loadSample(sampleId) {
    state.currentSampleId = sampleId;
    state.currentFile = null;
    showLoading(true);

    try {
      const res = await fetch('/api/predict-sample', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sample_id: sampleId,
          confidence_threshold: state.confidenceThreshold,
          crop_margins: state.cropMargins,
          return_gradcam: true
        })
      });

      if (!res.ok) throw new Error(`Inference returned status ${res.status}`);
      const data = await res.json();
      applyInferenceResults(data);
    } catch (err) {
      console.error('Failed to load sample prediction:', err);
      alert(`Inference failed: ${err.message}`);
    } finally {
      showLoading(false);
    }
  }

  async function handleFileUpload(file) {
    state.currentFile = file;
    state.currentSampleId = null;
    document.querySelectorAll('.sample-card').forEach(c => {
      c.classList.remove('active', 'border-blue-700', 'bg-blue-50', 'ring-2', 'ring-blue-700');
    });
    showLoading(true);

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('confidence_threshold', state.confidenceThreshold.toString());
      formData.append('crop_margins', state.cropMargins.toString());
      formData.append('return_gradcam', 'true');

      const res = await fetch('/api/predict', {
        method: 'POST',
        body: formData
      });

      if (!res.ok) throw new Error(`Inference returned status ${res.status}`);
      const data = await res.json();
      applyInferenceResults(data);
    } catch (err) {
      console.error('File inference failed:', err);
      alert(`File inference failed: ${err.message}`);
    } finally {
      showLoading(false);
    }
  }

  function applyInferenceResults(data) {
    state.currentPrediction = data;
    dom.viewportPlaceholder.classList.add('opacity-0');
    setTimeout(() => dom.viewportPlaceholder.classList.add('hidden'), 250);

    // Reset clinician action status on new case
    state.clinicianDetermination = 'Pending Review';
    dom.clinicianActionStatus.textContent = 'Pending Review';
    dom.clinicianActionStatus.className = 'text-[11px] font-bold px-2 py-0.5 rounded-full bg-slate-200 text-slate-700 border border-slate-300';
    dom.reportClinicianStatus.textContent = 'Pending Attending Review';
    dom.reportClinicianStatus.className = 'text-slate-700 bg-slate-200 px-3 py-1 rounded-full border border-slate-300 font-semibold';

    // Load base radiograph
    const baseImg = new Image();
    baseImg.onload = () => {
      state.currentImage = baseImg;
      fitImageToViewport();
      
      // Load colored heatmap overlay
      if (data.heatmap_png_base64) {
        const heatImg = new Image();
        heatImg.onload = () => {
          state.currentHeatmap = heatImg;
          renderCanvas();
        };
        heatImg.src = data.heatmap_png_base64;
      } else {
        state.currentHeatmap = null;
        renderCanvas();
      }
    };
    baseImg.src = data.image_base64;

    // Viewport HUD
    dom.overlayDimensions.textContent = `${data.image_dimensions.width} × ${data.image_dimensions.height}`;
    dom.overlayLatency.textContent = `${data.processing_time_ms} ms`;

    // Telemetry
    updateTelemetry(data);
  }

  function updateTelemetry(data) {
    const isPneu = data.is_pneumonia;
    const probPneu = data.probabilities ? data.probabilities.Pneumonia : (isPneu ? data.confidence : 1 - data.confidence);
    const pneuPercent = Math.round(probPneu * 100);

    // Primary Diagnostic Hero
    if (isPneu) {
      dom.diagnosisLabel.textContent = 'PNEUMONIA DETECTED';
      dom.diagnosisLabel.className = 'text-2xl font-black tracking-tight text-red-950';
      dom.diagnosisSubtext.textContent = 'Focal or diffuse consolidation indicated with elevated AI confidence.';
      dom.triageHero.className = 'p-5 border-b border-red-400 bg-red-100 transition-colors duration-300';
      dom.riskMeterBar.className = 'h-full rounded-full transition-all duration-500 bg-red-700';
    } else {
      dom.diagnosisLabel.textContent = 'NORMAL (CLEAR LUNGS)';
      dom.diagnosisLabel.className = 'text-2xl font-black tracking-tight text-emerald-950';
      dom.diagnosisSubtext.textContent = 'No evidence of acute focal pulmonary consolidation or infiltrate.';
      dom.triageHero.className = 'p-5 border-b border-emerald-400 bg-emerald-100 transition-colors duration-300';
      dom.riskMeterBar.className = 'h-full rounded-full transition-all duration-500 bg-emerald-700';
    }

    dom.probPneumonia.textContent = `${pneuPercent}%`;
    dom.riskMeterBar.style.width = `${pneuPercent}%`;

    // Borderline Uncertainty Alert
    if (data.is_borderline) {
      dom.borderlineBanner.classList.remove('hidden');
    } else {
      dom.borderlineBanner.classList.add('hidden');
    }

    // Dual-Backbone Gauges
    const breakdown = data.breakdown || {};
    const resPneu = breakdown['ResNet-50'] ? breakdown['ResNet-50'].Pneumonia : probPneu;
    const densePneu = breakdown['DenseNet-121'] ? breakdown['DenseNet-121'].Pneumonia : probPneu;

    const resPercent = Math.round(resPneu * 100);
    const densePercent = Math.round(densePneu * 100);

    dom.resScore.textContent = `${resPercent}% Pneumonia`;
    dom.resBar.style.width = `${resPercent}%`;

    dom.denseScore.textContent = `${densePercent}% Pneumonia`;
    dom.denseBar.style.width = `${densePercent}%`;

    // Consensus Badge
    const resVote = resPneu >= data.threshold_used;
    const denseVote = densePneu >= data.threshold_used;
    if (resVote === denseVote) {
      dom.consensusBadge.textContent = '100% Unanimous Agreement';
      dom.consensusBadge.className = 'text-xs font-extrabold px-3 py-0.5 rounded-full bg-emerald-100 text-emerald-950 border border-emerald-600';
    } else {
      dom.consensusBadge.textContent = 'Divergent Ensemble Vote';
      dom.consensusBadge.className = 'text-xs font-extrabold px-3 py-0.5 rounded-full bg-amber-100 text-amber-950 border border-amber-600';
    }

    // Marker Status
    dom.orientationStatus.textContent = data.crop_margins_applied ? 'Cropped & Suppressed' : 'Standard Window';
    dom.orientationStatus.className = data.crop_margins_applied ? 'font-black text-blue-900' : 'font-bold text-slate-900';

    // Attention Focus
    if (isPneu) {
      dom.attentionFocus.textContent = probPneu > 0.85 ? 'Segmental Consolidation' : 'Focal Infiltrate';
      dom.attentionFocus.className = 'font-black text-red-950';
    } else {
      dom.attentionFocus.textContent = 'Bilateral Clear Parenchyma';
      dom.attentionFocus.className = 'font-black text-emerald-950';
    }
  }

  function showLoading(show) {
    if (show) {
      dom.loadingOverlay.classList.remove('hidden');
    } else {
      dom.loadingOverlay.classList.add('hidden');
    }
  }

  // ================= CLINICAL REPORT MODAL =================
  function openReportModal() {
    if (!state.currentPrediction) {
      alert('Please select or upload a chest radiograph before generating a clinical report.');
      return;
    }
    const d = state.currentPrediction;
    dom.reportTime.textContent = new Date().toLocaleString();
    dom.reportPatientId.textContent = state.currentSampleId ? state.currentSampleId.toUpperCase() : 'UPLOADED-PATIENT-CASE';
    dom.reportCutoff.textContent = `${Math.round(d.threshold_used * 100)}%`;
    dom.reportLatency.textContent = `${d.processing_time_ms} ms`;
    
    dom.reportImpression.textContent = d.is_pneumonia 
      ? `POSITIVE: Pulmonary consolidation indicated with ${Math.round(d.confidence * 100)}% confidence. Grad-CAM localized focal lung opacities.`
      : `NEGATIVE: Clear lung fields with ${Math.round(d.confidence * 100)}% confidence. No diagnostic evidence of acute pneumonia.`;

    const bd = d.breakdown || {};
    dom.reportResNetProb.textContent = bd['ResNet-50'] ? `${Math.round(bd['ResNet-50'].Pneumonia * 100)}% Pneumonia Probability` : '--';
    dom.reportDenseNetProb.textContent = bd['DenseNet-121'] ? `${Math.round(bd['DenseNet-121'].Pneumonia * 100)}% Pneumonia Probability` : '--';
    dom.reportEnsembleProb.textContent = `${Math.round(d.confidence * 100)}% (${d.label})`;

    dom.reportModal.classList.remove('hidden');
    refreshIcons();
  }

  function closeReportModal() {
    dom.reportModal.classList.add('hidden');
  }

  // Initialize
  document.addEventListener('DOMContentLoaded', init);
})();
