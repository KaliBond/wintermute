// Neural Nations — nav mobile toggle
(function () {
    var btn  = document.getElementById('nav-hamburger');
    var menu = document.getElementById('nav-mobile');
    if (!btn || !menu) return;

    btn.addEventListener('click', function () {
        var open = menu.classList.toggle('open');
        btn.classList.toggle('open', open);
        btn.setAttribute('aria-label', open ? 'Close menu' : 'Open menu');
    });

    // Close on any mobile link click
    menu.querySelectorAll('a').forEach(function (a) {
        a.addEventListener('click', function () {
            menu.classList.remove('open');
            btn.classList.remove('open');
            btn.setAttribute('aria-label', 'Open menu');
        });
    });

    // Close on outside click
    document.addEventListener('click', function (e) {
        if (!btn.contains(e.target) && !menu.contains(e.target)) {
            menu.classList.remove('open');
            btn.classList.remove('open');
            btn.setAttribute('aria-label', 'Open menu');
        }
    });
})();
