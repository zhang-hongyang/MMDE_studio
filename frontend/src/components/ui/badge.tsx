import type { HTMLAttributes, ReactNode } from "react";
import { cn } from "../../lib/utils";

type Variant = "default" | "accent" | "success" | "danger" | "warning" | "outline";

const variants: Record<Variant, string> = {
  default: "bg-primary/15 text-primary",
  accent: "bg-accent/15 text-accent",
  success: "bg-success/15 text-success",
  danger: "bg-danger/15 text-danger",
  warning: "bg-warning/15 text-warning",
  outline: "border border-border text-fg-muted",
};

export function Badge({
  className,
  variant = "default",
  ...props
}: HTMLAttributes<HTMLSpanElement> & { variant?: Variant; children?: ReactNode }) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-[6px] px-1.5 py-0.5 text-[12px] font-medium leading-4",
        variants[variant],
        className,
      )}
      {...props}
    />
  );
}
