// Optional browser acquisition attribution is disabled alongside analytics.
// Keep sign-in callers inert; do not read, submit, or delete legacy storage.
export function capture() {}
export const firstTouchProps = () => ({});
export async function report() {}
