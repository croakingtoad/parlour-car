/**
 * Service Worker for The Author Library PWA
 * Implements app shell caching strategy with explicit WebSocket/API exclusions
 */

const CACHE_NAME = 'author-library-v1';
const CACHE_VERSION = '1.0.0';

// Static assets to cache (app shell)
const STATIC_CACHE_URLS = [
  '/',
  '/public/icons/pwa-192x192.png',
  '/public/icons/pwa-512x512.png',
  '/public/manifest.webmanifest',
];

// Patterns to NEVER cache (WebSocket, API, dynamic content)
const EXCLUDE_PATTERNS = [
  /^\/socket\.io\//,      // WebSocket connections
  /^\/project\//,         // Chainlit project routes (dynamic)
  /^\/api\//,             // API endpoints
  /^\/ws\//,              // WebSocket endpoints
  /^\/avatars\//,         // User-uploaded avatars
  /^\/files\//,           // User-uploaded files
];

/**
 * Check if a URL should be excluded from caching
 */
function shouldExclude(url) {
  const pathname = new URL(url).pathname;
  return EXCLUDE_PATTERNS.some(pattern => pattern.test(pathname));
}

/**
 * Install event - cache static app shell assets
 */
self.addEventListener('install', (event) => {
  console.log('[SW] Installing service worker version', CACHE_VERSION);

  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log('[SW] Caching app shell assets');
      return cache.addAll(STATIC_CACHE_URLS);
    }).then(() => {
      console.log('[SW] App shell cached successfully');
      // Force the waiting service worker to become the active service worker
      return self.skipWaiting();
    })
  );
});

/**
 * Activate event - clean up old caches
 */
self.addEventListener('activate', (event) => {
  console.log('[SW] Activating new service worker');

  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => {
          if (cacheName !== CACHE_NAME) {
            console.log('[SW] Deleting old cache:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    }).then(() => {
      console.log('[SW] Service worker activated, claiming clients');
      // Take control of all pages immediately
      return self.clients.claim();
    })
  );
});

/**
 * Fetch event - cache-first for static assets, network-only for excluded patterns
 */
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = request.url;

  // NEVER intercept WebSocket, API, or dynamic content
  if (shouldExclude(url)) {
    // Let these requests go directly to the network without caching
    return;
  }

  // For all other requests, try cache first, then network
  event.respondWith(
    caches.match(request).then((cachedResponse) => {
      if (cachedResponse) {
        console.log('[SW] Serving from cache:', url);
        return cachedResponse;
      }

      // Not in cache, fetch from network
      return fetch(request).then((networkResponse) => {
        // Only cache successful GET requests
        if (
          request.method === 'GET' &&
          networkResponse &&
          networkResponse.status === 200 &&
          networkResponse.type === 'basic'
        ) {
          // Cache static assets (JS, CSS, images)
          if (
            url.match(/\.(js|css|png|jpg|jpeg|svg|gif|woff|woff2|ttf|eot|ico)$/) ||
            url.includes('/assets/')
          ) {
            const responseToCache = networkResponse.clone();
            caches.open(CACHE_NAME).then((cache) => {
              console.log('[SW] Caching new asset:', url);
              cache.put(request, responseToCache);
            });
          }
        }

        return networkResponse;
      }).catch((error) => {
        console.error('[SW] Fetch failed:', error);

        // If offline and requesting navigation, return cached index
        if (request.mode === 'navigate') {
          return caches.match('/').then((cachedIndex) => {
            if (cachedIndex) {
              console.log('[SW] Offline: serving cached index page');
              return cachedIndex;
            }
          });
        }

        throw error;
      });
    })
  );
});

/**
 * Message event - handle commands from the registration script
 */
self.addEventListener('message', (event) => {
  if (event.data && event.data.type === 'SKIP_WAITING') {
    console.log('[SW] Received SKIP_WAITING message');
    self.skipWaiting();
  }
});

console.log('[SW] Service worker script loaded, version:', CACHE_VERSION);
