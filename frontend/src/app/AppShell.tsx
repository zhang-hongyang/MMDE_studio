import { NavLink, useLocation, useParams } from "react-router-dom";
import {
  LayoutDashboard,
  Telescope,
  Boxes,
  SlidersHorizontal,
  BarChart3,
  ListChecks,
  Boxes as LogoIcon,
} from "lucide-react";
import { useRegistryStore } from "../stores/registry";
import { useThemeStore } from "../stores/theme";
import { Switch } from "../components/ui/switch";
import { Separator } from "../components/ui/separator";
import { Tooltip } from "../components/ui/tooltip";
import { cn } from "../lib/utils";

const NAV = [
  { to: "/", label: "Overview", icon: LayoutDashboard, end: true },
  { to: "/explore", label: "Explorer", icon: Telescope, end: false },
  { to: "/scene", label: "Scene", icon: Boxes, end: false },
  { to: "/controls", label: "Controls", icon: SlidersHorizontal, end: true },
  { to: "/metrics", label: "Metrics", icon: BarChart3, end: true },
  { to: "/tasks", label: "Tasks", icon: ListChecks, end: true },
];

function pageTitle(pathname: string): string {
  if (pathname === "/") return "Overview";
  if (pathname.startsWith("/explore")) return "Explorer";
  if (pathname.startsWith("/scene")) return "Scene";
  if (pathname.startsWith("/controls")) return "Controls";
  if (pathname.startsWith("/metrics")) return "Metrics";
  if (pathname.startsWith("/tasks")) return "Tasks";
  return "MMDE Studio";
}

export function AppShell({ children }: { children: React.ReactNode }) {
  const location = useLocation();
  const params = useParams();
  const theme = useThemeStore((s) => s.theme);
  const toggle = useThemeStore((s) => s.toggle);
  const registry = useRegistryStore((s) => s.registry);

  const crumb =
    params.dataset && params.split ? (
      <span className="text-fg-muted">
        {" "}
        / <span className="font-mono">{params.dataset}</span> /{" "}
        <span className="font-mono">{params.split}</span>
      </span>
    ) : null;

  return (
    <div className="flex h-full min-h-0">
      {/* sidebar — 240px */}
      <nav className="flex w-[240px] shrink-0 flex-col border-r border-border bg-card">
        <div className="flex h-14 items-center gap-2 border-b border-border px-3">
          <LogoIcon className="h-5 w-5 text-primary" />
          <span className="text-[16px] font-semibold tracking-tight">MMDE Studio</span>
        </div>
        <div className="flex flex-col gap-0.5 p-2">
          {NAV.map((item) => (
            <NavLink
              key={item.label}
              to={item.to}
              end={item.end}
              className={({ isActive }) =>
                cn(
                  "relative flex h-9 items-center gap-2 rounded-[6px] px-3 text-[13px]",
                  "transition-colors duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                  isActive
                    ? "border-l-2 border-l-primary bg-primary/10 text-fg"
                    : "border-l-2 border-l-transparent text-fg-muted hover:bg-card-2 hover:text-fg",
                )
              }
            >
              <item.icon className="h-4 w-4 shrink-0" />
              {item.label}
            </NavLink>
          ))}
        </div>
        <div className="mt-auto p-3 text-[12px] text-fg-muted">
          <Separator className="mb-2" />
          <span className="num">
            {registry ? `${registry.datasets.length} 个数据集` : "后端未连接"}
          </span>
        </div>
      </nav>

      <div className="flex min-w-0 flex-1 flex-col">
        {/* header — 56px */}
        <header className="flex h-14 shrink-0 items-center gap-3 border-b border-border bg-card px-4">
          <h1 className="text-[16px] font-semibold">
            {pageTitle(location.pathname)}
            {crumb}
          </h1>
          <div className="ml-auto flex items-center gap-3">
            <span className="num hidden text-[12px] text-fg-muted sm:inline">
              {new Date().toLocaleDateString("zh-CN")}
            </span>
            <Tooltip.Root>
              <Tooltip.Trigger asChild>
                <label className="flex cursor-pointer items-center gap-2 text-[12px] text-fg-muted">
                  <span aria-hidden={theme === "dark"}>暗</span>
                  <Switch
                    checked={theme === "light"}
                    onCheckedChange={toggle}
                    aria-label="切换暗色/亮色主题"
                  />
                  <span aria-hidden={theme === "light"}>亮</span>
                </label>
              </Tooltip.Trigger>
              <Tooltip.Content>切换主题（写入 localStorage）</Tooltip.Content>
            </Tooltip.Root>
          </div>
        </header>
        <div className="min-h-0 flex-1">{children}</div>
      </div>
    </div>
  );
}
