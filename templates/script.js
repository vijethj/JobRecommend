window.addEventListener('scroll', function() {
    var parallax = document.querySelector('.parallax');
    var scrollPosition = window.pageYOffset;
    parallax.style.transform = 'translateY(' + scrollPosition * 0.5 + 'px)';
});
