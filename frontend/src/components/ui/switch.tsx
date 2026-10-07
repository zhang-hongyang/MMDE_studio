import * as React from "react";
import * as SwitchPrimitive from "@radix-ui/react-switch";
import { cn } from "../../lib/utils";

export const Switch = React.forwardRef<
  React.ElementRef<typeof SwitchPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof SwitchPrimitive.Root>
>(({ className, ...props }, ref) => (
  <SwitchPrimitive.Root
    ref={ref}
    className={cn(
      "inline-flex h-[18px] w-[34px] shrink-0 cursor-pointer items-center rounded-full border border-transparent",
      "transition-[background-color] duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
      "disabled:pointer-events-none disabled:opacity-50 data-[state=checked]:bg-primary data-[state=unchecked]:bg-border",
      className,
    )}
    {...props}
  >
    <SwitchPrimitive.Thumb
      className={cn(
        "block h-3.5 w-3.5 translate-x-[2px] rounded-full bg-on-primary shadow-sm",
        "transition-[transform] duration-150 data-[state=checked]:translate-x-[17px] data-[state=unchecked]:translate-x-[2px]",
      )}
    />
  </SwitchPrimitive.Root>
));
Switch.displayName = SwitchPrimitive.Root.displayName;
