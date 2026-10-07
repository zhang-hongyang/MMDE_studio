import { forwardRef, type ButtonHTMLAttributes } from "react";
import { cn } from "../../lib/utils";

type Variant = "default" | "secondary" | "outline" | "ghost" | "danger";
type Size = "sm" | "md" | "lg";

const variants: Record<Variant, string> = {
  default: "bg-primary text-on-primary hover:opacity-90 border border-transparent",
  secondary: "bg-card-2 text-fg border border-border hover:bg-border/60",
  outline: "bg-transparent text-fg border border-border hover:bg-card-2",
  ghost: "bg-transparent text-fg hover:bg-card-2",
  danger: "bg-danger text-white border border-transparent hover:opacity-90",
};

const sizes: Record<Size, string> = {
  sm: "h-[28px] px-2 text-[12px]",
  md: "h-[32px] px-3 text-[13px]",
  lg: "h-[40px] px-4 text-[14px]",
};

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = "default", size = "md", type = "button", ...props }, ref) => (
    <button
      ref={ref}
      type={type}
      className={cn(
        "inline-flex items-center justify-center gap-1.5 rounded-[6px] font-medium",
        "transition-[transform,opacity,background-color] duration-150 select-none",
        "disabled:pointer-events-none disabled:opacity-50",
        variants[variant],
        sizes[size],
        className,
      )}
      {...props}
    />
  ),
);
Button.displayName = "Button";
