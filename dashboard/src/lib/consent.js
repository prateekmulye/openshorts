// Optional browser analytics is unavailable. Legacy consent records included
// forced grants, so none may enable a tracker. Necessary storage stays untouched.
export const getConsent = () => ({ necessary: true, analytics: false, marketing: false });

// Keep the boot call inert until a separately reviewed tracker can stop safely.
export const applyConsent = () => false;

// Open the storage information panel without threading props through the app.
const OPEN_EVENT = 'openshorts:consent-open';
export const openConsentManager = () => {
  try { window.dispatchEvent(new CustomEvent(OPEN_EVENT)); } catch (_) { /* ignore */ }
};
export const onConsentOpenRequest = (fn) => {
  window.addEventListener(OPEN_EVENT, fn);
  return () => window.removeEventListener(OPEN_EVENT, fn);
};
