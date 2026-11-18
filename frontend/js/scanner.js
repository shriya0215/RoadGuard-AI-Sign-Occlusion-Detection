/**
 * RoadGuard AI - AI Road Sign Scanner Controller
 * Handles image selection, drag-and-drop, sample loading, real CNN inference,
 * and safety policy alert rendering.
 */

window.RoadGuardScanner = (function () {
  let selectedFile = null;
  let selectedBase64 = null;

  document.addEventListener('DOMContentLoaded', initScanner);

  function initScanner() {
    const dropzone = document.getElementById('scanner-dropzone');
    const fileInput = document.getElementById('scanner-file-input');
    const analyzeBtn = document.getElementById('scanner-analyze-btn');

    if (!dropzone || !fileInput) return;

    // File input change
    fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) {
        handleFileSelect(e.target.files[0]);
      }
    });

    // Drag & Drop events
    ['dragenter', 'dragover'].forEach(eventName => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
      });
    });

    ['dragleave', 'drop'].forEach(eventName => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
      });
    });

    dropzone.addEventListener('drop', (e) => {
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleFileSelect(e.dataTransfer.files[0]);
      }
    });

    // Analyze Button
    if (analyzeBtn) {
      analyzeBtn.addEventListener('click', runInference);
    }

    // Sample Buttons
    document.querySelectorAll('.scanner-sample-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const classId = btn.getAttribute('data-class-id');
        loadSampleImage(classId);
      });
    });
  }

  function handleFileSelect(file) {
    if (!file.type.match('image.*')) {
      alert('Please upload a valid image file (PNG, JPG, JPEG, WEBP, BMP).');
      return;
    }
    selectedFile = file;
    selectedBase64 = null;

    const reader = new FileReader();
    reader.onload = (e) => {
      displayPreview(e.target.result);
      enableAnalyzeBtn();
    };
    reader.readAsDataURL(file);
  }

  function displayPreview(dataUrl) {
    const previewImg = document.getElementById('scanner-preview-img');
    const placeholder = document.getElementById('scanner-preview-placeholder');
    if (previewImg) {
      previewImg.src = dataUrl;
      previewImg.style.display = 'block';
    }
    if (placeholder) {
      placeholder.style.display = 'none';
    }
  }

  function enableAnalyzeBtn() {
    const analyzeBtn = document.getElementById('scanner-analyze-btn');
    if (analyzeBtn) {
      analyzeBtn.disabled = false;
    }
  }

  async function loadSampleImage(classId) {
    try {
      const res = await fetch(`/dataset-sample/${classId}`);
      if (!res.ok) throw new Error('Could not load sample image');
      const blob = await res.blob();
      const file = new File([blob], `sample_class_${classId}.png`, { type: 'image/png' });
      handleFileSelect(file);
    } catch (err) {
      console.warn('[Scanner] Sample fetch fallback:', err);
    }
  }

  async function runInference() {
    if (!selectedFile && !selectedBase64) return;

    const analyzeBtn = document.getElementById('scanner-analyze-btn');
    const spinner = document.getElementById('scanner-spinner');
    const btnText = document.getElementById('scanner-btn-text');
    const errorBox = document.getElementById('scanner-error-box');

    if (errorBox) errorBox.style.display = 'none';
    if (analyzeBtn) analyzeBtn.disabled = true;
    if (spinner) spinner.style.display = 'inline-block';
    if (btnText) btnText.textContent = 'Processing CNN Inference...';

    const formData = new FormData();
    if (selectedFile) {
      formData.append('image', selectedFile);
    } else if (selectedBase64) {
      formData.append('image', selectedBase64);
    }

    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        body: formData
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.message || data.error || 'Inference failed');
      }

      displayResults(data);

      // Refresh global metrics
      if (typeof loadMetrics === 'function') {
        loadMetrics();
      }
    } catch (err) {
      console.error('[Scanner] Inference error:', err);
      if (errorBox) {
        errorBox.textContent = `Perception Subsystem Error: ${err.message}`;
        errorBox.style.display = 'block';
      }
    } finally {
      if (analyzeBtn) analyzeBtn.disabled = false;
      if (spinner) spinner.style.display = 'none';
      if (btnText) btnText.textContent = 'Analyze Road Sign';
    }
  }

  function displayResults(data) {
    const resultsContainer = document.getElementById('scanner-results-container');
    if (!resultsContainer) return;
    resultsContainer.style.display = 'block';

    // Sign Name & Confidence
    const signNameEl = document.getElementById('res-sign-name');
    const signConfEl = document.getElementById('res-sign-conf');
    const signCatEl = document.getElementById('res-sign-cat');
    const signFill = document.getElementById('res-sign-fill');

    if (signNameEl) signNameEl.textContent = data.traffic_sign.name;
    if (signConfEl) signConfEl.textContent = `${data.traffic_sign.confidence_percent}%`;
    if (signCatEl) signCatEl.textContent = data.traffic_sign.category;
    if (signFill) signFill.style.width = `${data.traffic_sign.confidence_percent}%`;

    // Occlusion Status & Confidence
    const occStatusEl = document.getElementById('res-occ-status');
    const occConfEl = document.getElementById('res-occ-conf');
    const occFill = document.getElementById('res-occ-fill');
    const occProbClear = document.getElementById('res-prob-clear');
    const occProbOcc = document.getElementById('res-prob-occluded');

    if (occStatusEl) {
      occStatusEl.textContent = data.occlusion.status;
      occStatusEl.className = `badge-status ${data.occlusion.is_occluded ? 'badge-warning' : 'badge-success'}`;
    }
    if (occConfEl) occConfEl.textContent = `${data.occlusion.confidence_percent}%`;
    if (occFill) occFill.style.width = `${data.occlusion.confidence_percent}%`;
    if (occProbClear) occProbClear.textContent = `${(data.occlusion.probabilities.clear * 100).toFixed(1)}%`;
    if (occProbOcc) occProbOcc.textContent = `${(data.occlusion.probabilities.occluded * 100).toFixed(1)}%`;

    // Top Candidates
    const candidatesList = document.getElementById('res-candidates-list');
    if (candidatesList && data.traffic_sign.top_candidates) {
      candidatesList.innerHTML = data.traffic_sign.top_candidates.map(c => `
        <div style="margin-bottom: 8px;">
          <div style="display:flex; justify-content:space-between; font-size:12px; margin-bottom:3px;">
            <span>${c.name}</span>
            <strong style="font-family:var(--font-mono); color:var(--accent-cyan);">${(c.confidence * 100).toFixed(1)}%</strong>
          </div>
          <div class="meter-bar">
            <div class="meter-fill meter-cyan" style="width: ${(c.confidence * 100).toFixed(1)}%"></div>
          </div>
        </div>
      `).join('');
    }

    // Module 5: Safety Policy Decision Banner
    const safetyBanner = document.getElementById('res-safety-banner');
    const safetyIcon = document.getElementById('res-safety-icon');
    const safetyTitle = document.getElementById('res-safety-title');
    const safetyRec = document.getElementById('res-safety-rec');
    const safetyDisc = document.getElementById('res-safety-disclaimer');

    if (safetyBanner && data.safety) {
      safetyBanner.className = `safety-banner ${data.safety.severity}`;
      if (safetyTitle) safetyTitle.textContent = data.safety.alert_title;
      if (safetyRec) safetyRec.textContent = data.safety.recommendation;
      if (safetyDisc) safetyDisc.textContent = data.safety.disclaimer;
    }
  }

  // Public API to set image from Occlusion Lab
  function setImageFromLab(base64Data) {
    selectedFile = null;
    selectedBase64 = base64Data;
    displayPreview(base64Data);
    enableAnalyzeBtn();

    // Switch to scanner tab
    const scannerTab = document.querySelector('.nav-link[data-target="scanner"]');
    if (scannerTab) scannerTab.click();

    // Run inference immediately
    setTimeout(runInference, 300);
  }

  return {
    setImageFromLab,
    handleFileSelect
  };
})();
