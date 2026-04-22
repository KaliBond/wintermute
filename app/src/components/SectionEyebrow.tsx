interface SectionEyebrowProps {
  text: string;
  color?: string;
  className?: string;
}

export function SectionEyebrow({ text, color = 'text-sage', className = '' }: SectionEyebrowProps) {
  return (
    <span className={`accent-label ${color} ${className}`}>
      {text}
    </span>
  );
}
