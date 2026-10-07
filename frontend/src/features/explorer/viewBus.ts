// Tiny pub/sub used to mirror viewport A's camera into viewport B
// when "sync view" is enabled.

export interface ViewState {
  position: [number, number, number];
  target: [number, number, number];
}

let latest: ViewState | null = null;
const subs = new Set<(v: ViewState) => void>();

export const viewBus = {
  publish(v: ViewState) {
    latest = v;
    subs.forEach((fn) => fn(v));
  },
  subscribe(fn: (v: ViewState) => void): () => void {
    subs.add(fn);
    return () => {
      subs.delete(fn);
    };
  },
  latest: () => latest,
};
