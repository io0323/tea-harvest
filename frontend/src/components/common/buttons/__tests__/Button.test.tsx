/**
 * ボタンコンポーネントのテストスイート
 * @jest-environment jsdom
 */
import { render, screen, fireEvent } from '@testing-library/react';
import { Button } from '../Button';

describe('Button Component', () => {
  /**
   * ボタンの基本的なレンダリングテスト
   */
  it('正しくレンダリングされること', () => {
    render(<Button>テストボタン</Button>);
    expect(screen.getByRole('button')).toBeInTheDocument();
    expect(screen.getByText('テストボタン')).toBeInTheDocument();
  });

  /**
   * ボタンのクリックイベントテスト
   */
  it('クリックイベントが正しく発火すること', () => {
    const mockOnClick = jest.fn();
    render(<Button onClick={mockOnClick}>クリックテスト</Button>);
    
    const button = screen.getByRole('button');
    fireEvent.click(button);
    
    expect(mockOnClick).toHaveBeenCalledTimes(1);
  });

  /**
   * ボタンの無効状態テスト
   */
  it('無効状態が正しく機能すること', () => {
    render(<Button disabled>無効ボタン</Button>);
    
    const button = screen.getByRole('button');
    expect(button).toBeDisabled();
    expect(button).toHaveAttribute('disabled');
  });

  /**
   * ボタンのバリアントテスト
   */
  it('異なるバリアントが正しく適用されること', () => {
    const { rerender } = render(<Button variant="primary">プライマリー</Button>);
    expect(screen.getByRole('button')).toHaveClass('bg-primary');

    rerender(<Button variant="secondary">セカンダリー</Button>);
    expect(screen.getByRole('button')).toHaveClass('bg-secondary');
  });
}); 