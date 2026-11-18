/**
 * RoadGuard AI - Chart.js Visualizations & Performance Metrics Controller
 */

window.RoadGuardCharts = (function () {
  let distChart = null;
  let lossChart = null;
  let accChart = null;

  document.addEventListener('DOMContentLoaded', () => {
    initDistributionChart();
  });

  /**
   * Initializes the Clear vs Occluded Doughnut Chart on Dashboard.
   */
  function initDistributionChart() {
    const ctx = document.getElementById('chart-occlusion-distribution');
    if (!ctx) return;

    distChart = new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: ['Clear Signs', 'Occluded Signs'],
        datasets: [{
          data: [60, 40],
          backgroundColor: ['#22C55E', '#F59E0B'],
          borderColor: '#0D1B2A',
          borderWidth: 3,
          hoverOffset: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'bottom',
            labels: {
              color: '#94A3B8',
              font: { family: 'Inter', size: 12 },
              padding: 16
            }
          },
          tooltip: {
            backgroundColor: '#071426',
            borderColor: '#00D9FF',
            borderWidth: 1,
            titleColor: '#F8FAFC',
            bodyColor: '#94A3B8'
          }
        },
        cutout: '72%'
      }
    });
  }

  function updateDistribution(clearCount, occludedCount) {
    if (!distChart) return;
    distChart.data.datasets[0].data = [clearCount, occludedCount];
    distChart.update();
  }

  /**
   * Loads training history JSON and renders Training/Validation curves for Module 6.
   */
  async function loadPerformanceCharts() {
    try {
      const res = await fetch('/api/history');
      if (!res.ok) return;
      const data = await res.json();
      if (!data.available) return;

      const hist = data.history;
      const epochs = Array.from({ length: hist.loss.length }, (_, i) => `Ep ${i + 1}`);

      // 1. Loss Chart
      const lossCtx = document.getElementById('chart-loss-history');
      if (lossCtx) {
        if (lossChart) lossChart.destroy();
        lossChart = new Chart(lossCtx, {
          type: 'line',
          data: {
            labels: epochs,
            datasets: [
              {
                label: 'Train Total Loss',
                data: hist.loss,
                borderColor: '#00D9FF',
                backgroundColor: 'rgba(0, 217, 255, 0.1)',
                borderWidth: 2,
                tension: 0.3
              },
              {
                label: 'Val Total Loss',
                data: hist.val_loss || [],
                borderColor: '#2563EB',
                borderWidth: 2,
                borderDash: [5, 5],
                tension: 0.3
              },
              {
                label: 'Sign Loss',
                data: hist.sign_output_loss || [],
                borderColor: '#F59E0B',
                borderWidth: 1.5,
                tension: 0.3
              },
              {
                label: 'Occlusion Loss',
                data: hist.occlusion_output_loss || [],
                borderColor: '#EF4444',
                borderWidth: 1.5,
                tension: 0.3
              }
            ]
          },
          options: getCommonChartOptions('Loss Value')
        });
      }

      // 2. Accuracy Chart
      const accCtx = document.getElementById('chart-accuracy-history');
      if (accCtx) {
        if (accChart) accChart.destroy();
        accChart = new Chart(accCtx, {
          type: 'line',
          data: {
            labels: epochs,
            datasets: [
              {
                label: 'Train Occlusion Acc',
                data: hist.occlusion_output_accuracy || [],
                borderColor: '#00D9FF',
                borderWidth: 2,
                tension: 0.3
              },
              {
                label: 'Val Occlusion Acc',
                data: hist.val_occlusion_output_accuracy || [],
                borderColor: '#38BDF8',
                borderWidth: 2,
                borderDash: [5, 5],
                tension: 0.3
              },
              {
                label: 'Train Sign Acc',
                data: hist.sign_output_accuracy || [],
                borderColor: '#22C55E',
                borderWidth: 2,
                tension: 0.3
              },
              {
                label: 'Val Sign Acc',
                data: hist.val_sign_output_accuracy || [],
                borderColor: '#10B981',
                borderWidth: 2,
                borderDash: [5, 5],
                tension: 0.3
              }
            ]
          },
          options: getCommonChartOptions('Accuracy', 0, 1.0)
        });
      }
    } catch (err) {
      console.warn('[Charts] History fetch error:', err);
    }
  }

  function getCommonChartOptions(yTitle, minY = null, maxY = null) {
    return {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          grid: { color: 'rgba(148, 163, 184, 0.08)' },
          ticks: { color: '#94A3B8', font: { family: 'Inter', size: 11 } }
        },
        y: {
          min: minY,
          max: maxY,
          title: { display: true, text: yTitle, color: '#94A3B8', font: { size: 12 } },
          grid: { color: 'rgba(148, 163, 184, 0.08)' },
          ticks: { color: '#94A3B8', font: { family: 'Inter', size: 11 } }
        }
      },
      plugins: {
        legend: {
          position: 'top',
          labels: { color: '#F8FAFC', font: { family: 'Inter', size: 12 }, padding: 12 }
        },
        tooltip: {
          backgroundColor: '#071426',
          borderColor: '#00D9FF',
          borderWidth: 1,
          titleColor: '#F8FAFC',
          bodyColor: '#94A3B8'
        }
      }
    };
  }

  return {
    updateDistribution,
    loadPerformanceCharts
  };
})();

// Hook called when user switches to Model Performance tab
function loadPerformanceData() {
  if (window.RoadGuardCharts) {
    window.RoadGuardCharts.loadPerformanceCharts();
  }
}
