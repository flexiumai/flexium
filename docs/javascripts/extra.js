// Open external links in new tab (only for links going to different domains)
function setupExternalLinks() {
  var currentHost = window.location.hostname;

  document.querySelectorAll("a[href]").forEach(function(link) {
    // Skip if already has target set
    if (link.hasAttribute("target")) return;

    // Use the link's hostname property (browser resolves relative URLs)
    var linkHost = link.hostname;

    // If the link goes to a different domain, open in new tab
    if (linkHost && linkHost !== currentHost) {
      link.setAttribute("target", "_blank");
      link.setAttribute("rel", "noopener");
    }
  });
}

// Run on initial load
document.addEventListener("DOMContentLoaded", setupExternalLinks);

// Re-run when MkDocs Material does instant navigation (SPA-style page changes)
if (typeof document$ !== "undefined") {
  document$.subscribe(setupExternalLinks);
}
