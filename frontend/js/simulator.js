/**
 * RoadGuard AI - Occlusion Lab & Comparison Simulator Controller
 * Powers Module 3 (Synthetic Occlusion Simulator) & Module 4 (Before vs After Analysis)
 */

window.RoadGuardSimulator = (function () {
  let labSelectedFile = null;
  let labCurrentType = 'leaves';
  let labIntensity = 0.50;
  let labLastOccludedBase64 = null;

  // Comparison State
  let compSelectedFile = null;
  let compCurrentType = 'leaves';
  let compIntensity = 0.50;

  document.addEventListener('DOMContentLoaded', () => {
    initOcclusionLab();
    initComparisonModule();
  });

  /* -------------------- MODULE 3: OCCLUSION LAB -------------------- */
  function initOcclusionLab() {
    const fileInput = document.getElementById('lab-file-input');
    const intensitySlider = document.getElementById('lab-intensity-slider');
    const intensityVal = document.getElementById('lab-intensity-val');
    const generateBtn = document.getElementById('lab-generate-btn');
    const sendToScannerBtn = document.getElementById('lab-send-to-scanner-btn');

    // Type Selector Chips
    document.querySelectorAll('.lab-type-chip').forEach(chip => {
      chip.addEventListener('click', () => {
        document.querySelectorAll('.lab-type-chip').forEach(c => c.classList.remove('selected'));
        chip.classList.add('selected');
        labCurrentType = chip.getAttribute('data-type');
      });
    });

    // Intensity Slider
    if (intensitySlider) {
      intensitySlider.addEventListener('input', (e) => {
        labIntensity = parseFloat(e.target.value);
        if (intensityVal) {
          intensityVal.textContent = `${Math.round(labIntensity * 100)}%`;
        }
      });
    }

    // File Input
    if (fileInput) {
      fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
          labSelectedFile = e.target.files[0];
          const reader = new FileReader();
          reader.onload = (ev) => {
            const origImg = document.getElementById('lab-orig-img');
            const placeholder = document.getElementById('lab-orig-placeholder');
            if (origImg) {
              origImg.src = ev.target.result;
              origImg.style.display = 'block';
            }
            if (placeholder) placeholder.style.display = 'none';
            if (generateBtn) generateBtn.disabled = false;
          };
          reader.readAsDataURL(labSelectedFile);
        }
      });
    }

    // Sample Picker Chips
    document.querySelectorAll('.lab-sample-chip').forEach(chip => {
      chip.addEventListener('click', async () => {
        const classId = chip.getAttribute('data-class-id');
        try {
          const res = await fetch(`/dataset-sample/${classId}`);
          if (!res.ok) throw new Error('Sample fetch failed');
          const blob = await res.blob();
          labSelectedFile = new File([blob], `sample_${classId}.png`, { type: 'image/png' });
          const reader = new FileReader();
          reader.onload = (ev) => {
            const origImg = document.getElementById('lab-orig-img');
            const placeholder = document.getElementById('lab-orig-placeholder');
            if (origImg) {
              origImg.src = ev.target.result;
              origImg.style.display = 'block';
            }
            if (placeholder) placeholder.style.display = 'none';
            if (generateBtn) generateBtn.disabled = false;
          };
          reader.readAsDataURL(labSelectedFile);
        } catch (err) {
          console.warn('[Lab] Sample error:', err);
        }
      });
    });

    // Generate Button
    if (generateBtn) {
      generateBtn.addEventListener('click', generateLabOcclusion);
    }

    // Send to Scanner Button
    if (sendToScannerBtn) {
      sendToScannerBtn.addEventListener('click', () => {
        if (labLastOccludedBase64 && window.RoadGuardScanner) {
          window.RoadGuardScanner.setImageFromLab(labLastOccludedBase64);
        }
      });
    }
  }

  async function generateLabOcclusion() {
    if (!labSelectedFile) return;

    const generateBtn = document.getElementById('lab-generate-btn');
    const sendToScannerBtn = document.getElementById('lab-send-to-scanner-btn');
    const spinner = document.getElementById('lab-spinner');
    const btnText = document.getElementById('lab-btn-text');

    if (generateBtn) generateBtn.disabled = true;
    if (spinner) spinner.style.display = 'inline-block';
    if (btnText) btnText.textContent = 'Synthesizing Texture...';

    const formData = new FormData();
    formData.append('image', labSelectedFile);
    formData.append('occlusion_type', labCurrentType);
    formData.append('intensity', labIntensity);

    try {
      const res = await fetch('/api/simulate-occlusion', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.message || 'Simulation failed');

      labLastOccludedBase64 = data.occluded_image;

      const occImg = document.getElementById('lab-occ-img');
      const placeholder = document.getElementById('lab-occ-placeholder');
      if (occImg) {
        occImg.src = data.occluded_image;
        occImg.style.display = 'block';
      }
      if (placeholder) placeholder.style.display = 'none';

      // Metadata labels
      const metaType = document.getElementById('lab-res-type');
      const metaInt = document.getElementById('lab-res-intensity');
      if (metaType) metaType.textContent = data.occlusion_type.toUpperCase();
      if (metaInt) metaInt.textContent = `${Math.round(data.intensity * 100)}% Coverage`;

      if (sendToScannerBtn) sendToScannerBtn.disabled = false;
    } catch (err) {
      alert(`Simulation Error: ${err.message}`);
    } finally {
      if (generateBtn) generateBtn.disabled = false;
      if (spinner) spinner.style.display = 'none';
      if (btnText) btnText.textContent = 'Generate Occlusion';
    }
  }


  /* -------------------- MODULE 4: BEFORE VS AFTER COMPARISON -------------------- */
  function initComparisonModule() {
    const fileInput = document.getElementById('comp-file-input');
    const intensitySlider = document.getElementById('comp-intensity-slider');
    const intensityVal = document.getElementById('comp-intensity-val');
    const runBtn = document.getElementById('comp-run-btn');

    document.querySelectorAll('.comp-type-chip').forEach(chip => {
      chip.addEventListener('click', () => {
        document.querySelectorAll('.comp-type-chip').forEach(c => c.classList.remove('selected'));
        chip.classList.add('selected');
        compCurrentType = chip.getAttribute('data-type');
      });
    });

    if (intensitySlider) {
      intensitySlider.addEventListener('input', (e) => {
        compIntensity = parseFloat(e.target.value);
        if (intensityVal) {
          intensityVal.textContent = `${Math.round(compIntensity * 100)}%`;
        }
      });
    }

    if (fileInput) {
      fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
          compSelectedFile = e.target.files[0];
          if (runBtn) runBtn.disabled = false;
        }
      });
    }

    // Comparison Sample Chips
    document.querySelectorAll('.comp-sample-chip').forEach(chip => {
      chip.addEventListener('click', async () => {
        const classId = chip.getAttribute('data-class-id');
        try {
          const res = await fetch(`/dataset-sample/${classId}`);
          if (!res.ok) throw new Error('Sample fetch failed');
          const blob = await res.blob();
          compSelectedFile = new File([blob], `comp_${classId}.png`, { type: 'image/png' });
          if (runBtn) runBtn.disabled = false;
          // Trigger comparison automatically
          runComparisonAnalysis();
        } catch (err) {
          console.warn('[Comparison] Sample error:', err);
        }
      });
    });

    if (runBtn) {
      runBtn.addEventListener('click', runComparisonAnalysis);
    }
  }

  async function runComparisonAnalysis() {
    if (!compSelectedFile) {
      alert('Please select or upload a road sign image to analyze.');
      return;
    }

    const runBtn = document.getElementById('comp-run-btn');
    const spinner = document.getElementById('comp-spinner');
    const btnText = document.getElementById('comp-btn-text');
    const resultsContainer = document.getElementById('comp-results-container');

    if (runBtn) runBtn.disabled = true;
    if (spinner) spinner.style.display = 'inline-block';
    if (btnText) btnText.textContent = 'Running Dual Inferences...';

    const formData = new FormData();
    formData.append('image', compSelectedFile);
    formData.append('occlusion_type', compCurrentType);
    formData.append('intensity', compIntensity);

    try {
      const res = await fetch('/api/compare', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.message || 'Comparison failed');

      if (resultsContainer) resultsContainer.style.display = 'block';

      // Render Original Side
      document.getElementById('comp-orig-img').src = data.original.image;
      document.getElementById('comp-orig-sign').textContent = data.original.sign_name;
      document.getElementById('comp-orig-conf').textContent = `${data.original.sign_confidence}%`;
      document.getElementById('comp-orig-occ').textContent = `${data.original.is_occluded ? 'Occluded' : 'Clear'} (${data.original.occlusion_confidence}%)`;

      // Render Occluded Side
      document.getElementById('comp-occ-img').src = data.occluded.image;
      document.getElementById('comp-occ-sign').textContent = data.occluded.sign_name;
      document.getElementById('comp-occ-conf').textContent = `${data.occluded.sign_confidence}%`;
      document.getElementById('comp-occ-occ').textContent = `${data.occluded.is_occluded ? 'Occluded' : 'Clear'} (${data.occluded.occlusion_confidence}%)`;

      // Render Comparison Metrics
      const deltaEl = document.getElementById('comp-delta-conf');
      const delta = data.comparison.confidence_delta_percent;
      if (deltaEl) {
        deltaEl.textContent = `${delta > 0 ? '+' : ''}${delta}%`;
        deltaEl.style.color = delta < -10 ? 'var(--status-danger)' : 'var(--accent-cyan)';
      }

      const explanationEl = document.getElementById('comp-explanation-text');
      if (explanationEl) {
        explanationEl.textContent = data.comparison.explanation;
      }

      const alertBox = document.getElementById('comp-adversarial-alert');
      if (alertBox) {
        if (data.comparison.class_swapped) {
          alertBox.style.display = 'block';
          alertBox.innerHTML = `<strong>⚠️ CRITICAL PERCEPTION HAZARD:</strong> The occlusion pattern caused an adversarial misclassification! Ground truth is compromised.`;
        } else {
          alertBox.style.display = 'none';
        }
      }
    } catch (err) {
      alert(`Comparison Error: ${err.message}`);
    } finally {
      if (runBtn) runBtn.disabled = false;
      if (spinner) spinner.style.display = 'none';
      if (btnText) btnText.textContent = 'Run Comparative Analysis';
    }
  }

  return {
    generateLabOcclusion,
    runComparisonAnalysis
  };
})();
