import { useEffect } from "react";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "./AppShell";
import { OverviewPage } from "../pages/OverviewPage";
import { PlaceholderPage } from "../pages/PlaceholderPage";
import { ExplorerPage } from "../features/explorer/ExplorerPage";
import { ScenePage } from "../features/scene/ScenePage";
import { ControlsPage } from "../features/controls/ControlsPage";
import { MetricsPage } from "../features/metrics/MetricsPage";
import { TasksPage } from "../features/tasks/TasksPage";
import { useRegistryStore } from "../stores/registry";
import { Tooltip } from "../components/ui/tooltip";

/** /explore and /scene redirect to the first dataset/split from the registry. */
function FirstDatasetRedirect({ sub }: { sub: string }) {
  const registry = useRegistryStore((s) => s.registry);
  const d = registry?.datasets[0];
  const sp = d?.splits[0]?.name;
  if (d && sp) return <Navigate to={`/${sub}/${d.name}/${sp}`} replace />;
  return (
    <PlaceholderPage
      title={sub === "explore" ? "Explorer" : sub === "scene" ? "Scene" : "Controls"}
    />
  );
}

export function App() {
  const load = useRegistryStore((s) => s.load);
  useEffect(() => {
    void load();
  }, [load]);

  return (
    <Tooltip.Provider delayDuration={300}>
      <BrowserRouter>
        <AppShell>
          <Routes>
            <Route path="/" element={<OverviewPage />} />
            <Route path="/explore" element={<FirstDatasetRedirect sub="explore" />} />
            <Route path="/explore/:dataset/:split" element={<ExplorerPage />} />
            <Route path="/scene" element={<FirstDatasetRedirect sub="scene" />} />
            <Route path="/scene/:dataset/:split" element={<ScenePage />} />
            <Route path="/controls" element={<FirstDatasetRedirect sub="controls" />} />
            <Route path="/controls/:dataset/:split" element={<ControlsPage />} />
            <Route path="/metrics" element={<MetricsPage />} />
            <Route path="/tasks" element={<TasksPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </AppShell>
      </BrowserRouter>
    </Tooltip.Provider>
  );
}
