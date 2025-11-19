// main.js: small helpers used by pages
console.log('FruitAI UI loaded');

// Provide a small utility to safely replace result area on fetch responses that return HTML
async function fetchAndReplace(url, options, container){
  const res = await fetch(url, options);
  if(!res.ok) throw new Error('Network error');
  const html = await res.text();
  container.innerHTML = html;
}

// Graceful error reporting for fetch
window.fetchJson = async function(url, opts){
  const res = await fetch(url, opts);
  if(!res.ok) throw new Error('Request failed');
  return res.json();
}
