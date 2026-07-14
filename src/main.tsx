// Safeguard against missing chrome.runtime.getManifest in sandboxed environments
(function() {
  try {
    if (typeof window !== 'undefined') {
      var chr = (window as any).chrome || {};
      if (!(window as any).chrome) {
        try {
          (window as any).chrome = chr;
        } catch (e) {}
      }
      var rt = chr.runtime || {};
      if (!chr.runtime) {
        try {
          chr.runtime = rt;
        } catch (e) {}
      }
      if (typeof rt.getManifest !== 'function') {
        try {
          rt.getManifest = function() {
            return { version: "1.0.0" };
          };
        } catch (e) {}
      }
    }
  } catch (e) {
    console.warn('Unable to patch chrome.runtime.getManifest:', e);
  }
})();

import {StrictMode} from 'react';
import {createRoot} from 'react-dom/client';
import App from './App.tsx';
import './index.css';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
