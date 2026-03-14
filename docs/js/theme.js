/* cogdiff — theme detection + Chart.js color management */

(function () {
  'use strict';

  function getSystemTheme() {
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }

  function getSavedTheme() {
    try { return localStorage.getItem('cogdiff-theme'); } catch (e) { return null; }
  }

  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    try { localStorage.setItem('cogdiff-theme', theme); } catch (e) { /* noop */ }
    var btn = document.getElementById('themeToggle');
    if (btn) btn.textContent = theme === 'dark' ? '\u2600' : '\u263E';
  }

  var initial = getSavedTheme() || 'light';
  applyTheme(initial);

  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function (e) {
    if (!getSavedTheme()) applyTheme(e.matches ? 'dark' : 'light');
  });

  window.cdToggleTheme = function () {
    var current = document.documentElement.getAttribute('data-theme') || getSystemTheme();
    applyTheme(current === 'dark' ? 'light' : 'dark');
    if (window.cdRebuildCharts) window.cdRebuildCharts();
  };

  window.cdIsDark = function () {
    return document.documentElement.getAttribute('data-theme') === 'dark';
  };

  window.cdColors = function () {
    var dark = cdIsDark();
    return {
      navy:     dark ? '#c5cfe0' : '#1a1a2e',
      accent:   dark ? '#7ba0e0' : '#537ec5',
      red:      dark ? '#ef5350' : '#c0392b',
      green:    dark ? '#66bb6a' : '#27ae60',
      orange:   dark ? '#ffa726' : '#e67e22',
      purple:   dark ? '#ab47bc' : '#8e44ad',
      teal:     dark ? '#26a69a' : '#16a085',
      dark:     dark ? '#90a4ae' : '#2c3e50',
      deepOrange: dark ? '#ff7043' : '#d35400',
      muted:    dark ? '#3a4050' : '#d1d5db',
      text:     dark ? '#d4d8e0' : '#1f2937',
      textSec:  dark ? '#9ca3af' : '#6b7280',
      grid:     dark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.08)',
      bg:       dark ? '#1a1e2c' : '#ffffff',
      // Cohort colors
      nlsy79:   dark ? '#7ba0e0' : '#537ec5',
      nlsy97:   dark ? '#ffa726' : '#e67e22',
      cnlsy:    dark ? '#66bb6a' : '#27ae60',
      palette:  dark
        ? ['#7ba0e0', '#ffa726', '#66bb6a', '#ef5350', '#ab47bc', '#90a4ae', '#26a69a', '#ff7043']
        : ['#537ec5', '#e67e22', '#27ae60', '#c0392b', '#8e44ad', '#2c3e50', '#16a085', '#d35400']
    };
  };

  window.cdChartDefaults = function () {
    if (typeof Chart === 'undefined') return;
    try {
      var c = cdColors();
      Chart.defaults.font.family = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
      Chart.defaults.font.size = 13;
      Chart.defaults.color = c.textSec;
      Chart.defaults.borderColor = c.grid;
      if (Chart.defaults.plugins && Chart.defaults.plugins.legend && Chart.defaults.plugins.legend.labels) {
        Chart.defaults.plugins.legend.labels.color = c.textSec;
      }
      if (Chart.defaults.plugins && Chart.defaults.plugins.tooltip) {
        Chart.defaults.plugins.tooltip.backgroundColor = c.navy;
        Chart.defaults.plugins.tooltip.titleColor = '#fff';
        Chart.defaults.plugins.tooltip.bodyColor = '#e0e0e0';
      }
    } catch (e) { /* continue */ }
  };
})();
