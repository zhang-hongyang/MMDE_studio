import { Construction } from "lucide-react";

export function PlaceholderPage({ title }: { title: string }) {
  return (
    <div className="flex h-full items-center justify-center p-4">
      <div className="card flex w-full max-w-md flex-col items-center gap-2 py-8 text-center">
        <Construction className="h-6 w-6 text-warning" />
        <h1 className="text-[16px] font-semibold">{title}</h1>
        <p className="text-[13px] text-fg-muted">Coming in Milestone 2</p>
      </div>
    </div>
  );
}
