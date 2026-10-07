import * as React from "react";
import * as TooltipPrimitive from "@radix-ui/react-tooltip";
import { cn } from "../../lib/utils";

const Provider = TooltipPrimitive.Provider;

const Root = TooltipPrimitive.Root;

const Trigger = TooltipPrimitive.Trigger;

const Content = React.forwardRef<
  React.ElementRef<typeof TooltipPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof TooltipPrimitive.Content>
>(({ className, sideOffset = 4, ...props }, ref) => (
  <TooltipPrimitive.Portal>
    <TooltipPrimitive.Content
      ref={ref}
      sideOffset={sideOffset}
      className={cn(
        "z-50 max-w-64 rounded-[8px] border border-border bg-card px-2 py-1 text-[12px] text-fg shadow-md",
        "animate-in fade-in-0 zoom-in-95 data-[state=closed]:animate-out data-[state=closed]:fade-out-0",
        className,
      )}
      {...props}
    />
  </TooltipPrimitive.Portal>
));
Content.displayName = TooltipPrimitive.Content.displayName;

export const Tooltip = { Provider, Root, Trigger, Content };
