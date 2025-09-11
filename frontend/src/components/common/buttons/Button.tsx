/**
 * 共通ボタンコンポーネント
 * アプリケーション全体で使用する標準的なボタンコンポーネント
 */
import { ButtonHTMLAttributes, ReactNode } from 'react';
import { cn } from '@/lib/utils';

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  /**
   * ボタンの内容
   */
  children: ReactNode;
  
  /**
   * ボタンのバリアント（スタイルの種類）
   */
  variant?: 'primary' | 'secondary' | 'outline' | 'ghost';
  
  /**
   * ボタンのサイズ
   */
  size?: 'sm' | 'md' | 'lg';
  
  /**
   * カスタムクラス名
   */
  className?: string;
}

export const Button = ({
  children,
  variant = 'primary',
  size = 'md',
  className,
  disabled,
  ...props
}: ButtonProps) => {
  return (
    <button
      className={cn(
        // ベースのスタイル
        'inline-flex items-center justify-center rounded-md font-medium transition-colors',
        // バリアント別のスタイル
        {
          'bg-primary text-white hover:bg-primary/90': variant === 'primary',
          'bg-secondary text-secondary-foreground hover:bg-secondary/80': variant === 'secondary',
          'border border-input bg-background hover:bg-accent': variant === 'outline',
          'hover:bg-accent hover:text-accent-foreground': variant === 'ghost',
        },
        // サイズ別のスタイル
        {
          'h-9 px-4 text-sm': size === 'sm',
          'h-10 px-6 text-base': size === 'md',
          'h-11 px-8 text-lg': size === 'lg',
        },
        // 無効状態のスタイル
        {
          'opacity-50 cursor-not-allowed': disabled,
        },
        className
      )}
      disabled={disabled}
      {...props}
    >
      {children}
    </button>
  );
}; 