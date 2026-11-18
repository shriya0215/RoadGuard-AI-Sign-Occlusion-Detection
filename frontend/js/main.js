/**
 * RoadGuard AI - Main Dashboard Controller & Navigation
 */

// Global Application State
window.RoadGuard = {
  currentTab: 'dashboard',
  isModelTrained: false,
  metrics: null,
  health: null
};

document.addEventListener('DOMContentLoaded', () => {
  initNavigation();
  checkSystemHealth();
  loadMetrics();
  
  // Refresh metrics periodically
  setInterval(checkSystemHealth, 15000);
});

/**
 * Initializes sidebar tab navigation and URL hash syncing.
 */
function initNavigation() {
  const navLinks = document.querySelectorAll('.nav-link');
  const panes = document.querySelectorAll('.tab-pane');
  const pageTitle = document.getElementById('current-page-title');

  function switchTab(targetId) {
    navLinks.forEach(link => {
      if (link.getAttribute('data-target') === targetId) {
        link.classList.add('active');
      } else {
        link.classList.remove('active');
      }
    });

    panes.forEach(pane => {
      if (pane.id === `view-${targetId}`) {
        pane.classList.add('active');
      } else {
        pane.classList.remove('active');
      }
    });

    window.RoadGuard.currentTab = targetId;
    if (pageTitle) {
      const activeLink = document.querySelector(`.nav-link[data-target="${targetId}"]`);
      if (activeLink) {
        pageTitle.textContent = activeLink.querySelector('.nav-text').textContent;
      }
    }

    // Module-specific hooks
    if (targetId === 'dashboard') {
      loadMetrics();
    } else if (targetId === 'performance') {
      loadPerformanceData();
    }
  }

  navLinks.forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      const target = link.getAttribute('data-target');
      if (target) {
        switchTab(target);
        window.location.hash = target;
      }
    });
  });

  // Check URL hash on page load
  if (window.location.hash) {
    const hashTarget = window.location.hash.substring(1);
    if (document.getElementById(`view-${hashTarget}`)) {
      switchTab(hashTarget);
    }
  }
}

/**
 * Polls backend health status.
 */
async function checkSystemHealth() {
  try {
    const res = await fetch('/api/health');
    if (!res.ok) throw new Error('Health check failed');
    const data = await res.json();
    window.RoadGuard.health = data;
    window.RoadGuard.isModelTrained = data.model_trained;

    const statusDot = document.getElementById('system-status-dot');
    const statusText = document.getElementById('system-status-text');
    const modelBadge = document.getElementById('topbar-model-status');

    if (data.model_trained) {
      if (statusDot) { statusDot.className = 'pulse-dot'; }
      if (statusText) { statusText.textContent = 'Vision Stack Active'; }
      if (modelBadge) {
        modelBadge.className = 'badge-status badge-success';
        modelBadge.innerHTML = '<span class="status-indicator"></span> Dual-Head CNN Online';
      }
    } else {
      if (statusDot) { statusDot.className = 'pulse-dot warning'; }
      if (statusText) { statusText.textContent = 'Model Not Trained'; }
      if (modelBadge) {
        modelBadge.className = 'badge-status badge-warning';
        modelBadge.textContent = 'Model Not Trained';
      }
    }
  } catch (err) {
    console.warn('[RoadGuard] Health check error:', err);
  }
}

/**
 * Loads actual metrics from the backend and updates KPI cards.
 */
async function loadMetrics() {
  try {
    const res = await fetch('/api/metrics');
    if (!res.ok) throw new Error('Failed to load metrics');
    const data = await res.json();
    window.RoadGuard.metrics = data;

    if (!data.model_trained) {
      document.querySelectorAll('.kpi-real-value').forEach(el => {
        el.textContent = 'N/A';
        el.title = 'Model Not Trained';
      });
      return;
    }

    // Update KPIs with real evaluated metrics
    const signAccEl = document.getElementById('kpi-sign-accuracy');
    const occPrecEl = document.getElementById('kpi-occ-precision');
    const occRecallEl = document.getElementById('kpi-occ-recall');
    const occF1El = document.getElementById('kpi-occ-f1');
    const totalProcEl = document.getElementById('kpi-total-processed');
    const totalOccEl = document.getElementById('kpi-total-occluded');

    if (signAccEl) signAccEl.textContent = `${(data.sign_accuracy * 100).toFixed(1)}%`;
    if (occPrecEl) occPrecEl.textContent = `${(data.occlusion_precision * 100).toFixed(1)}%`;
    if (occRecallEl) occRecallEl.textContent = `${(data.occlusion_recall * 100).toFixed(1)}%`;
    if (occF1El) occF1El.textContent = `${(data.occlusion_f1 * 100).toFixed(1)}%`;
    if (totalProcEl) totalProcEl.textContent = data.session_processed || data.total_test_samples || 0;
    if (totalOccEl) totalOccEl.textContent = data.total_occluded_detected || 0;

    // Render recent session history table
    renderRecentHistory(data.recent_history || []);

    // Update distribution chart if available
    if (window.RoadGuardCharts && window.RoadGuardCharts.updateDistribution) {
      window.RoadGuardCharts.updateDistribution(
        data.confusion_matrix?.true_negative + data.confusion_matrix?.false_positive || 1,
        data.confusion_matrix?.true_positive + data.confusion_matrix?.false_negative || 1
      );
    }
  } catch (err) {
    console.error('[RoadGuard] Failed to fetch metrics:', err);
  }
}

/**
 * Renders the recent prediction history table in Dashboard.
 */
function renderRecentHistory(history) {
  const tbody = document.getElementById('recent-history-tbody');
  if (!tbody) return;

  if (!history || history.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="6" style="text-align:center; color: var(--text-dim); padding: 24px;">
          No images analyzed in this session yet. Upload a sign in the AI Scanner to see live detections.
        </td>
      </tr>`;
    return;
  }

  tbody.innerHTML = history.slice().reverse().map(item => {
    let badgeClass = 'badge-success';
    if (item.severity === 'warning') badgeClass = 'badge-warning';
    if (item.severity === 'danger') badgeClass = 'badge-danger';

    return `
      <tr>
        <td><strong>#${item.id}</strong></td>
        <td><span style="font-weight:600; color:var(--accent-cyan);">${item.sign_name}</span></td>
        <td><span style="font-family:var(--font-mono);">${item.sign_confidence}%</span></td>
        <td>
          <span class="badge-status ${item.occlusion_status === 'Occluded' ? 'badge-warning' : 'badge-success'}">
            ${item.occlusion_status} (${item.occlusion_confidence}%)
          </span>
        </td>
        <td><span class="badge-status ${badgeClass}">${item.status}</span></td>
      </tr>`;
  }).join('');
}
