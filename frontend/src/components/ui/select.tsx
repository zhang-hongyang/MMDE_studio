import * as React from "react";
import * as SelectPrimitive from "@radix-ui/react-select";
import { Check, ChevronDown, ChevronUp } from "lucide-react";
import { cn } from "../../lib/utils";

const Root = SelectPrimitive.Root;
const Value = SelectPrimitive.Value;
const Group = SelectPrimitive.Group;

const Trigger = React.forwardRef<
  React.ElementRef<typeof SelectPrimitive.Trigger>,
  React.ComponentPropsWithoutRef<typeof SelectPrimitive.Trigger>
>(({ className, children, ...props }, ref) => (
  <SelectPrimitive.Trigger
    ref={ref}
    className={cn(
      "flex h-[32px] items-center justify-between gap-1.5 rounded-[6px] border border-border bg-card-2 px-2",
      "text-[13px] text-fg hover:bg-border/50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
      "disabled:pointer-events-none disabled:opacity-50 data-[state=open]:ring-2 data-[state=open]:ring-ring [&>span]:truncate",
      className,
    )}
    {...props}
  >
    {children}
    <SelectPrimitive.Icon asChild>
      <ChevronDown className="h-3.5 w-3.5 shrink-0 text-fg-muted" />
    </SelectPrimitive.Icon>
  </SelectPrimitive.Trigger>
));
Trigger.displayName = SelectPrimitive.Trigger.displayName;

const Content = React.forwardRef<
  React.ElementRef<typeof SelectPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof SelectPrimitive.Content>
>(({ className, children, position = "popper", ...props }, ref) => (
  <SelectPrimitive.Portal>
    <SelectPrimitive.Content
      ref={ref}
      position={position}
      className={cn(
        "relative z-50 max-h-80 min-w-[8rem] overflow-hidden rounded-[8px] border border-border bg-card text-fg",
        "shadow-md data-[state=open]:animate-in data-[state=closed]:animate-out",
        "data-[side=bottom]:translate-y-1 data-[side=top]:-translate-y-1",
        position === "popper" && "data-[side=bottom]:mt-1 data-[side=top]:mb-1",
        className,
      )}
      {...props}
    >
      <SelectPrimitive.ScrollUpButton className="flex h-5 items-center justify-center">
        <ChevronUp className="h-3 w-3 text-fg-muted" />
      </SelectPrimitive.ScrollUpButton>
      <SelectPrimitive.Viewport
        className={cn(
          "p-1",
          position === "popper" &&
            "h-[var(--radix-select-trigger-height)] w-full min-w-[var(--radix-select-trigger-width)]",
        )}
      >
        {children}
      </SelectPrimitive.Viewport>
      <SelectPrimitive.ScrollDownButton className="flex h-5 items-center justify-center">
        <ChevronDown className="h-3 w-3 text-fg-muted" />
      </SelectPrimitive.ScrollDownButton>
    </SelectPrimitive.Content>
  </SelectPrimitive.Portal>
));
Content.displayName = SelectPrimitive.Content.displayName;

const Item = React.forwardRef<
  React.ElementRef<typeof SelectPrimitive.Item>,
  React.ComponentPropsWithoutRef<typeof SelectPrimitive.Item>
>(({ className, children, ...props }, ref) => (
  <SelectPrimitive.Item
    ref={ref}
    className={cn(
      "relative flex h-[28px] cursor-default items-center rounded-[6px] py-1 pl-2 pr-6 text-[13px] outline-none",
      "focus:bg-primary/15 focus:text-fg data-[disabled]:pointer-events-none data-[disabled]:opacity-50",
      className,
    )}
    {...props}
  >
    <span className="absolute right-1.5 flex h-3.5 w-3.5 items-center justify-center">
      <SelectPrimitive.ItemIndicator>
        <Check className="h-3.5 w-3.5 text-primary" />
      </SelectPrimitive.ItemIndicator>
    </span>
    <SelectPrimitive.ItemText>{children}</SelectPrimitive.ItemText>
  </SelectPrimitive.Item>
));
Item.displayName = SelectPrimitive.Item.displayName;

const Label = React.forwardRef<
  React.ElementRef<typeof SelectPrimitive.Label>,
  React.ComponentPropsWithoutRef<typeof SelectPrimitive.Label>
>(({ className, ...props }, ref) => (
  <SelectPrimitive.Label
    ref={ref}
    className={cn("px-2 py-1 text-[12px] font-medium text-fg-muted", className)}
    {...props}
  />
));
Label.displayName = SelectPrimitive.Label.displayName;

export const Select = { Root, Trigger, Content, Item, Value, Group, Label };
