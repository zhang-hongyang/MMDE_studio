import { create } from "zustand";
import { api, type Registry } from "../lib/api";

interface RegistryState {
  registry: Registry | null;
  loading: boolean;
  error: string | null;
  load: () => Promise<void>;
}

export const useRegistryStore = create<RegistryState>((set, get) => ({
  registry: null,
  loading: false,
  error: null,
  load: async () => {
    if (get().registry || get().loading) return;
    set({ loading: true, error: null });
    try {
      const registry = await api.registry();
      set({ registry, loading: false });
    } catch (e) {
      set({ loading: false, error: String(e) });
    }
  },
}));
