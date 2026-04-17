// Neural Nations — nav interactions
(function () {

    // ── Dropdown: click-toggle (works on touch + desktop) ────────
    document.querySelectorAll('.nav-dropdown').forEach(function (dd) {
        var toggle = dd.querySelector('.nav-dropdown-toggle');
        if (!toggle) return;

        toggle.addEventListener('click', function (e) {
            e.stopPropagation();
            var isOpen = dd.classList.contains('open');
            // Close all dropdowns first
            document.querySelectorAll('.nav-dropdown.open').forEach(function (other) {
                other.classList.remove('open');
            });
            if (!isOpen) dd.classList.add('open');
        });
    });

    // Close dropdown on outside click or Escape
    document.addEventListener('click', function () {
        document.querySelectorAll('.nav-dropdown.open').forEach(function (dd) {
            dd.classList.remove('open');
        });
    });
    document.addEventListener('keydown', function (e) {
        if (e.key === 'Escape') {
            document.querySelectorAll('.nav-dropdown.open').forEach(function (dd) {
                dd.classList.remove('open');
            });
        }
    });

    // ── Mobile hamburger ──────────────────────────────────────────
    var btn  = document.getElementById('nav-hamburger');
    var menu = document.getElementById('nav-mobile');
    if (btn && menu) {
        btn.addEventListener('click', function (e) {
            e.stopPropagation();
            var open = menu.classList.toggle('open');
            btn.classList.toggle('open', open);
            btn.setAttribute('aria-label', open ? 'Close menu' : 'Open menu');
        });

        menu.querySelectorAll('a').forEach(function (a) {
            a.addEventListener('click', function () {
                menu.classList.remove('open');
                btn.classList.remove('open');
                btn.setAttribute('aria-label', 'Open menu');
            });
        });
    }

})();
