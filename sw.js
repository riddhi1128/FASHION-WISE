self.addEventListener('install', e => {
    e.waitUntil(caches.open('fw-v1').then(c => c.addAll(['/', '/index.html'])));
    self.skipWaiting();
});
self.addEventListener('fetch', e => {
    if (e.request.method !== 'GET') return;
    e.respondWith(fetch(e.request).catch(() => caches.match(e.request)));
});