// Open external nav links in new tab
document.addEventListener("DOMContentLoaded", function() {
  document.querySelectorAll("nav a[href^='https://']").forEach(function(link) {
    link.setAttribute("target", "_blank");
    link.setAttribute("rel", "noopener");
  });
});
