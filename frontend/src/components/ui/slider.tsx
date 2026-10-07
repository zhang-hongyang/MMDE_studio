import * as React from "react";
import * as SliderPrimitive from "@radix-ui/react-slider";
import { cn } from "../../lib/utils";

const Root = React.forwardRef<
  React.ElementRef<typeof SliderPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof SliderPrimitive.Root>
>(({ className, ...props }, ref) => (
  <SliderPrimitive.Root
    ref={ref}
    // NOTE: no w-full here — in shrink-to-fit (inline-flex) toolbar layouts a
    // percentage width resolves against an indefinite containing block and the
    // track collapses to 0px, killing all pointer interaction. Callers pass an
    // explicit width (w-24 / w-28 / ...).
    className={cn(
      "relative flex h-[32px] touch-none select-none items-center",
      className,
    )}
    {...props}
  />
));
Root.displayName = SliderPrimitive.Root.displayName;

const Track = React.forwardRef<
  React.ElementRef<typeof SliderPrimitive.Track>,
  React.ComponentPropsWithoutRef<typeof SliderPrimitive.Track>
>(({ className, ...props }, ref) => (
  <SliderPrimitive.Track
    ref={ref}
    className={cn("relative h-1 w-full grow overflow-hidden rounded-full bg-border", className)}
    {...props}
  />
));
Track.displayName = SliderPrimitive.Track.displayName;

const Range = React.forwardRef<
  React.ElementRef<typeof SliderPrimitive.Range>,
  React.ComponentPropsWithoutRef<typeof SliderPrimitive.Range>
>(({ className, ...props }, ref) => (
  <SliderPrimitive.Range
    ref={ref}
    className={cn("absolute h-full bg-primary", className)}
    {...props}
  />
));
Range.displayName = SliderPrimitive.Range.displayName;

const Thumb = React.forwardRef<
  React.ElementRef<typeof SliderPrimitive.Thumb>,
  React.ComponentPropsWithoutRef<typeof SliderPrimitive.Thumb>
>(({ className, ...props }, ref) => (
  <SliderPrimitive.Thumb
    ref={ref}
    className={cn(
      "block h-3.5 w-3.5 rounded-full border-2 border-primary bg-card",
      "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
      className,
    )}
    {...props}
  />
));
Thumb.displayName = SliderPrimitive.Thumb.displayName;

export const Slider = { Root, Track, Range, Thumb };
