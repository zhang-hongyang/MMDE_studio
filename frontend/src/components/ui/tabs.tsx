import * as React from "react";
import * as TabsPrimitive from "@radix-ui/react-tabs";
import { cn } from "../../lib/utils";

const Root = TabsPrimitive.Root;

const List = React.forwardRef<
  React.ElementRef<typeof TabsPrimitive.List>,
  React.ComponentPropsWithoutRef<typeof TabsPrimitive.List>
>(({ className, ...props }, ref) => (
  <TabsPrimitive.List
    ref={ref}
    className={cn(
      "inline-flex h-[32px] items-center gap-0.5 rounded-[6px] border border-border bg-card-2 p-0.5",
      className,
    )}
    {...props}
  />
));
List.displayName = TabsPrimitive.List.displayName;

const Trigger = React.forwardRef<
  React.ElementRef<typeof TabsPrimitive.Trigger>,
  React.ComponentPropsWithoutRef<typeof TabsPrimitive.Trigger>
>(({ className, ...props }, ref) => (
  <TabsPrimitive.Trigger
    ref={ref}
    className={cn(
      "inline-flex h-[26px] items-center rounded-[6px] px-2.5 text-[13px] text-fg-muted",
      "transition-[background-color,color] duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
      "data-[state=active]:bg-card data-[state=active]:text-fg data-[state=active]:shadow-sm",
      className,
    )}
    {...props}
  />
));
Trigger.displayName = TabsPrimitive.Trigger.displayName;

const Content = React.forwardRef<
  React.ElementRef<typeof TabsPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof TabsPrimitive.Content>
>(({ className, ...props }, ref) => (
  <TabsPrimitive.Content ref={ref} className={cn("outline-none", className)} {...props} />
));
Content.displayName = TabsPrimitive.Content.displayName;

export const Tabs = { Root, List, Trigger, Content };
