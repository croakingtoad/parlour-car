/**
 * Service Worker Registration for The Author Library PWA
 * Handles registration, updates, and lifecycle management
 */

(function() {
  'use strict';

  // Only register on HTTPS or localhost
  const isSecureContext = location.protocol === 'https:' || location.hostname === 'localhost';
  const supportsServiceWorker = 'serviceWorker' in navigator;

  if (!supportsServiceWorker) {
    console.log('[PWA] Service workers not supported in this browser');
    return;
  }

  if (!isSecureContext) {
    console.log('[PWA] Service worker requires HTTPS (or localhost)');
    console.log('[PWA] Current protocol:', location.protocol);
    console.log('[PWA] For HTTPS testing, use Tailscale or ngrok');
    return;
  }

  console.log('[PWA] Initializing Progressive Web App features');

  /**
   * Register the service worker
   */
  function registerServiceWorker() {
    return navigator.serviceWorker
      .register('/public/sw.js', { scope: '/' })
      .then((registration) => {
        console.log('[PWA] Service worker registered successfully');
        console.log('[PWA] Scope:', registration.scope);

        // Check for updates on page load
        registration.update();

        // Set up update detection
        setupUpdateHandling(registration);

        // Check for updates periodically (every 5 minutes)
        setInterval(() => {
          console.log('[PWA] Checking for service worker updates');
          registration.update();
        }, 5 * 60 * 1000);

        return registration;
      })
      .catch((error) => {
        console.error('[PWA] Service worker registration failed:', error);
        throw error;
      });
  }

  /**
   * Handle service worker updates
   */
  function setupUpdateHandling(registration) {
    // When a new service worker is found and installing
    registration.addEventListener('updatefound', () => {
      const newWorker = registration.installing;
      console.log('[PWA] New service worker found, installing...');

      newWorker.addEventListener('statechange', () => {
        console.log('[PWA] Service worker state:', newWorker.state);

        if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
          // New service worker is installed but waiting to activate
          console.log('[PWA] New version available! Will activate on next page load.');

          // Optionally, notify the user about the update
          // You could show a toast/banner here asking user to refresh
          notifyUserOfUpdate();

          // Or automatically activate (be careful with this approach)
          // newWorker.postMessage({ type: 'SKIP_WAITING' });
        }
      });
    });

    // When the service worker takes control
    navigator.serviceWorker.addEventListener('controllerchange', () => {
      console.log('[PWA] Service worker controller changed');

      // Reload to get the latest version (only if not first load)
      if (window.pwaBecameActive) {
        console.log('[PWA] Reloading to activate new version');
        window.location.reload();
      }
    });

    window.pwaBecameActive = true;
  }

  /**
   * Notify user of available update
   */
  function notifyUserOfUpdate() {
    // Log to console (can be enhanced with UI notification)
    console.log('[PWA] 📦 App update available');
    console.log('[PWA] Refresh the page to get the latest version');

    // Optional: Show a subtle notification banner
    // This would require adding UI elements to the page
    try {
      const event = new CustomEvent('pwa-update-available', {
        detail: { message: 'A new version is available. Refresh to update.' }
      });
      window.dispatchEvent(event);
    } catch (e) {
      // Ignore if CustomEvent not supported
    }
  }

  /**
   * Check for updates when page becomes visible
   */
  function setupVisibilityChangeHandler() {
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'visible') {
        console.log('[PWA] Page visible, checking for updates');
        navigator.serviceWorker.ready.then((registration) => {
          registration.update();
        });
      }
    });
  }

  /**
   * Initialize PWA features when page loads
   */
  function init() {
    console.log('[PWA] Registering service worker');

    registerServiceWorker()
      .then((registration) => {
        console.log('[PWA] Registration successful');

        // Set up visibility change detection for updates
        setupVisibilityChangeHandler();

        // Check if already installed
        if (window.matchMedia('(display-mode: standalone)').matches) {
          console.log('[PWA] Running as installed PWA');
        }
      })
      .catch((error) => {
        console.error('[PWA] Registration error:', error);
      });
  }

  // Register when page loads
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    // DOM already loaded
    init();
  }

  // Expose PWA status for debugging
  window.pwaStatus = {
    isSecure: isSecureContext,
    supportsServiceWorker: supportsServiceWorker,
    isInstalled: window.matchMedia('(display-mode: standalone)').matches,
    checkForUpdates: () => {
      navigator.serviceWorker.ready.then((registration) => {
        registration.update();
      });
    }
  };

  console.log('[PWA] Registration script loaded');
})();
