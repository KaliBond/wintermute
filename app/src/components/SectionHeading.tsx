interface SectionHeadingProps {
  children: React.ReactNode;
  className?: string;
  as?: 'h1' | 'h2' | 'h3';
}

export function SectionHeading({ children, className = '', as: Tag = 'h2' }: SectionHeadingProps) {
  return (
    <Tag className={`font-serif text-parchment tracking-[-0.03em] leading-[1.1] ${className}`}>
      {children}
    </Tag>
  );
}
