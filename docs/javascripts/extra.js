// Open external links in new tab (only for links going outside docs.flexium.ai)
function setupExternalLinks() {
  // Only target links that go to external domains (not relative or same-site)
  document.querySelectorAll("a[href^='https://'], a[href^='http://']").forEach(function(link) {
    var href = link.getAttribute("href");
    // Skip if it's a link to the docs site itself
    if (href && !href.includes("docs.flexium.ai")) {
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
