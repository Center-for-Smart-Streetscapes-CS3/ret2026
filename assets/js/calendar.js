(() => {
  const calendar = document.querySelector('[data-calendar]');
  if (!calendar) return;

  const validYears = new Set(['2024', '2025', '2026']);
  const yearButtons = [...calendar.querySelectorAll('[data-year-button]')];
  const yearSections = [...calendar.querySelectorAll('[data-calendar-year]')];
  const filterButtons = [...calendar.querySelectorAll('[data-filter]')];
  const yearLabel = document.querySelector('[data-course-year]');
  const emptyMessage = calendar.querySelector('[data-calendar-empty]');
  const resultsStatus = calendar.querySelector('[data-results-status]');
  let activeFilter = 'All';
  let activeYear = '2026';

  const yearFromLocation = () => {
    const requested = new URL(window.location.href).searchParams.get('year');
    if (validYears.has(requested)) return requested;
    const hashMatch = window.location.hash.match(/^#y(24|25|26)/);
    return hashMatch ? `20${hashMatch[1]}` : '2026';
  };

  const activeSection = () => yearSections.find((section) => section.dataset.calendarYear === activeYear);

  const updateCounts = () => {
    const section = activeSection();
    const resources = section ? [...section.querySelectorAll('[data-resource-type]')] : [];

    filterButtons.forEach((button) => {
      const type = button.dataset.filter;
      const count = type === 'All'
        ? resources.length
        : resources.filter((resource) => resource.dataset.resourceType === type).length;
      const target = button.querySelector(`[data-filter-count="${type}"]`);
      if (target) target.textContent = String(count);
    });
  };

  const applyFilter = () => {
    const section = activeSection();
    if (!section) return;

    let visibleCount = 0;
    section.querySelectorAll('[data-calendar-week]').forEach((week) => {
      let weekCount = 0;

      week.querySelectorAll('[data-calendar-day]').forEach((day) => {
        let dayCount = 0;
        day.querySelectorAll('[data-resource-type]').forEach((resource) => {
          const visible = activeFilter === 'All' || resource.dataset.resourceType === activeFilter;
          resource.hidden = !visible;
          if (visible) dayCount += 1;
        });
        day.hidden = dayCount === 0;
        weekCount += dayCount;
      });

      week.hidden = weekCount === 0;
      visibleCount += weekCount;
    });

    filterButtons.forEach((button) => {
      button.setAttribute('aria-pressed', String(button.dataset.filter === activeFilter));
    });

    emptyMessage.hidden = visibleCount !== 0;
    resultsStatus.textContent = `${visibleCount} ${activeFilter === 'All' ? 'course entries' : activeFilter.toLowerCase() + ' entries'} shown for ${activeYear}.`;
  };

  const writeYearToUrl = (year, mode = 'push') => {
    const url = new URL(window.location.href);
    url.searchParams.set('year', year);
    const hashYear = url.hash.match(/^#y(24|25|26)/);
    if (hashYear && hashYear[1] !== year.slice(-2)) url.hash = '';
    history[mode === 'replace' ? 'replaceState' : 'pushState']({ year }, '', url);
  };

  const selectYear = (year, options = {}) => {
    if (!validYears.has(year)) return;
    activeYear = year;

    yearButtons.forEach((button) => {
      button.setAttribute('aria-pressed', String(button.dataset.yearButton === year));
    });
    yearSections.forEach((section) => {
      section.hidden = section.dataset.calendarYear !== year;
    });
    if (yearLabel) yearLabel.textContent = year;

    updateCounts();
    applyFilter();

    if (options.updateUrl) writeYearToUrl(year, options.historyMode);
  };

  yearButtons.forEach((button) => {
    button.addEventListener('click', () => {
      selectYear(button.dataset.yearButton, { updateUrl: true, historyMode: 'push' });
    });
  });

  filterButtons.forEach((button) => {
    button.addEventListener('click', () => {
      activeFilter = button.dataset.filter;
      applyFilter();
    });
  });

  window.addEventListener('popstate', () => selectYear(yearFromLocation()));

  selectYear(yearFromLocation(), { updateUrl: true, historyMode: 'replace' });

  if (window.location.hash) {
    const target = document.querySelector(window.location.hash);
    if (target instanceof HTMLDetailsElement) {
      target.open = true;
      requestAnimationFrame(() => target.scrollIntoView({ block: 'start' }));
    }
  }
})();
