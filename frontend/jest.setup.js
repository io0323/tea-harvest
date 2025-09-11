/**
 * Jestのグローバルセットアップファイル
 * テスト環境のグローバル設定とマッチャーの拡張を行います
 */
import '@testing-library/jest-dom';

// グローバルなテスト設定をここに追加
jest.setTimeout(10000); // タイムアウトを10秒に設定

// モックのマッチャーをカスタマイズする場合はここに追加
expect.extend({
  // カスタムマッチャーをここに追加
});

// テスト実行時のコンソールエラーを抑制
beforeAll(() => {
  console.error = (...args) => {
    if (
      args[0].includes('Warning: ReactDOM.render is no longer supported') ||
      args[0].includes('Warning: React.createFactory()')
    ) {
      return;
    }
    console.error(...args);
  };
}); 