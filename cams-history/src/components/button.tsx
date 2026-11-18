import { cva, type VariantProps } from "class-variance-authority";
import type { ButtonHTMLAttributes } from "react";
import { cn } from "@/lib/cn";

const buttonStyles = cva(
  "inline-flex h-11 items-center justify-center gap-2 rounded-sm px-4 text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40",
  {
    variants: {
      tone: {
        brass: "bg-brass text-canvas hover:bg-brass/90",
        ghost: "border border-line bg-transparent text-bone hover:border-mute",
        danger: "bg-danger text-canvas hover:bg-danger/90",
        quiet: "bg-transparent px-2 text-mute hover:text-bone",
      },
    },
    defaultVariants: { tone: "brass" },
  },
);

export function Button({
  tone,
  className,
  type = "button",
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & VariantProps<typeof buttonStyles>) {
  return <button type={type} className={cn(buttonStyles({ tone }), className)} {...props} />;
}
