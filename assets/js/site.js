(() => {
  const toggle = document.querySelector('[data-nav-toggle]');
  const nav = document.querySelector('[data-primary-nav]');

  if (toggle && nav) {
    const header = document.querySelector('[data-site-header]');
    const main = document.querySelector('main');
    const footer = document.querySelector('footer');

    const positionMenu = () => {
      if (!header) return;
      const top = Math.max(0, header.getBoundingClientRect().bottom);
      nav.style.setProperty('--mobile-nav-top', `${top}px`);
    };

    const setBackgroundInert = (inert) => {
      if (main) main.inert = inert;
      if (footer) footer.inert = inert;
    };

    const closeMenu = () => {
      toggle.setAttribute('aria-expanded', 'false');
      nav.removeAttribute('data-open');
      document.body.classList.remove('nav-open');
      setBackgroundInert(false);
    };

    toggle.addEventListener('click', () => {
      const open = toggle.getAttribute('aria-expanded') === 'true';
      if (!open) positionMenu();
      toggle.setAttribute('aria-expanded', String(!open));
      nav.toggleAttribute('data-open', !open);
      document.body.classList.toggle('nav-open', !open);
      setBackgroundInert(!open);
      if (!open) requestAnimationFrame(() => nav.querySelector('a')?.focus());
    });

    nav.addEventListener('click', (event) => {
      if (event.target.closest('a')) closeMenu();
    });

    document.addEventListener('keydown', (event) => {
      const open = toggle.getAttribute('aria-expanded') === 'true';
      if (event.key === 'Escape' && open) {
        closeMenu();
        toggle.focus();
      }
      if (event.key === 'Tab' && open) {
        const focusable = [toggle, ...nav.querySelectorAll('a')];
        const first = focusable[0];
        const last = focusable[focusable.length - 1];
        if (event.shiftKey && document.activeElement === first) {
          event.preventDefault();
          last.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
          event.preventDefault();
          first.focus();
        }
      }
    });

    window.matchMedia('(min-width: 960px)').addEventListener('change', (event) => {
      if (event.matches) closeMenu();
    });

    window.addEventListener('resize', () => {
      if (toggle.getAttribute('aria-expanded') === 'true') positionMenu();
    });
  }

  document.querySelectorAll('a[target="_blank"]').forEach((link) => {
    link.rel = 'noopener noreferrer';
  });
})();
