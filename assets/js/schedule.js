(() => {
  const schedule = document.querySelector('[data-schedule]');
  if (!schedule) return;

  const validYears = new Set(['2024', '2025', '2026']);
  const buttons = [...schedule.querySelectorAll('[data-schedule-year-button]')];
  const panels = [...schedule.querySelectorAll('[data-schedule-year]')];
  const yearLabel = document.querySelector('[data-schedule-course-year]');
  const dailyYearLabel = schedule.querySelector('[data-schedule-daily-year]');

  const yearFromLocation = () => {
    const requested = new URL(window.location.href).searchParams.get('year');
    return validYears.has(requested) ? requested : '2026';
  };

  const writeYearToUrl = (year, mode = 'push') => {
    const url = new URL(window.location.href);
    url.searchParams.set('year', year);
    history[mode === 'replace' ? 'replaceState' : 'pushState']({ year }, '', url);
  };

  const selectYear = (year, options = {}) => {
    if (!validYears.has(year)) return;
    buttons.forEach((button) => {
      button.setAttribute('aria-pressed', String(button.dataset.scheduleYearButton === year));
    });
    panels.forEach((panel) => {
      panel.hidden = panel.dataset.scheduleYear !== year;
    });
    if (yearLabel) yearLabel.textContent = year;
    if (dailyYearLabel) dailyYearLabel.textContent = year;
    if (options.updateUrl) writeYearToUrl(year, options.historyMode);
  };

  buttons.forEach((button) => {
    button.addEventListener('click', () => {
      selectYear(button.dataset.scheduleYearButton, { updateUrl: true, historyMode: 'push' });
    });
  });

  window.addEventListener('popstate', () => selectYear(yearFromLocation()));
  selectYear(yearFromLocation(), { updateUrl: true, historyMode: 'replace' });
})();
