// Optional browser analytics is disabled. The pinned SDK cannot safely cancel
// pending requests on revocation; clear() only clears identifiers.
// ponytail: inert exports preserve callers — restore only with reviewed teardown.
export function track() {}
export function identify() {}
export function reset() {}
