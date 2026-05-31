// Reading progress bar
const bar = document.getElementById('progress-bar');
window.addEventListener('scroll', () => {
  const doc = document.documentElement;
  const scrolled = doc.scrollTop;
  const total = doc.scrollHeight - doc.clientHeight;
  bar.style.width = `${(scrolled / total) * 100}%`;
});

// Smooth-highlight TOC link for current section
const sections = document.querySelectorAll('h2[id]');
const tocLinks = document.querySelectorAll('.toc a');

const observer = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      tocLinks.forEach(l => l.style.color = '');
      const active = document.querySelector(`.toc a[href="#${entry.target.id}"]`);
      if (active) active.style.color = 'var(--accent)';
    }
  });
}, { rootMargin: '-20% 0px -70% 0px' });

sections.forEach(s => observer.observe(s));
