// No consent prompt while there are no optional browser trackers to enable.
import { useEffect, useRef, useState } from 'react';
import { onConsentOpenRequest } from '../lib/consent';

export default function CookieBanner() {
  const [open, setOpen] = useState(false);
  const dialogRef = useRef(null);
  useEffect(() => {
    if (open) dialogRef.current.showModal();
    else if (dialogRef.current.open) dialogRef.current.close();
  }, [open]);
  useEffect(() => onConsentOpenRequest(() => setOpen(true)), []);

  return (
    <>
    <button type="button" className="btn-quiet fixed bottom-3 right-3 z-50 bg-paper" onClick={() => setOpen(true)}>
      Cookies and storage
    </button>
    <dialog
      ref={dialogRef}
      onClose={() => setOpen(false)}
      aria-label="Cookie and tracker preferences"
      className="fixed inset-x-0 bottom-0 top-auto m-0 w-full max-w-none bg-transparent p-3 sm:p-4 pb-[max(0.75rem,env(safe-area-inset-bottom))]"
    >
      <div className="card mx-auto max-w-3xl p-4 sm:p-5 space-y-4 shadow-2xl">
        <h2 className="font-display lowercase text-lg text-ink">Cookies and storage</h2>
        <p className="text-sm text-ink2">
          Optional browser analytics is disabled. We do not load audience measurement
          or advertising trackers in your browser, and there is nothing optional to enable.
        </p>
        <p className="text-sm text-ink2">
          Necessary storage remains available for your sign-in session, access to your
          clips, and app preferences.{' '}
          <a href="/privacy" className="underline hover:text-ink">Privacy policy</a>.
        </p>
        <button type="button" autoFocus className="btn-ghost" onClick={() => setOpen(false)}>Close</button>
      </div>
    </dialog>
    </>
  );
}
