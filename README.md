# 茶葉収穫時期予測システム

## 概要

気象データを基に、茶葉の最適な収穫時期を予測するWebアプリケーションです。
機械学習モデルを使用して、気温、湿度、降水量、日照時間などのデータから収穫の最適なタイミングを提案します。

## 技術スタック

### フロントエンド
- Next.js (v14.2.25)
- React (v18.2.0)
- TypeScript (v5.2.2)
- Tailwind CSS (v3.4.1)
- Shadcn/ui
- Clerk (認証)

### バックエンド
- Next.js API Routes
- Prisma (ORM)
- SQLite (開発環境)
- Supabase (本番環境予定)

## 主な機能

1. 気象データ分析
   - CSVまたはExcelファイルのアップロード
   - 気象データの自動解析
   - 収穫時期の予測

2. 予測履歴管理
   - 過去の予測結果の閲覧
   - 詳細な気象条件の確認
   - 予測精度の表示

3. ユーザー管理
   - Clerkによる認証
   - ユーザーごとの予測履歴
   - 設定のカスタマイズ

## 開発環境のセットアップ

1. リポジトリのクローン
```bash
git clone https://github.com/io0323/tea-harvest.git
cd tea-harvest
```

2. 依存関係のインストール
```bash
npm install
```

3. 環境変数の設定
```bash
cp .env.example .env.local
# .env.localファイルに必要な環境変数を設定
```

4. 開発サーバーの起動
```bash
npm run dev
```

## 環境変数

以下の環境変数が必要です：

```env
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=your_publishable_key
CLERK_SECRET_KEY=your_secret_key
DATABASE_URL=your_database_url
```

## ディレクトリ構造

```
tea-harvest/
├── app/                    # Next.jsアプリケーション
│   ├── (auth)/            # 認証関連ページ
│   ├── dashboard/         # ダッシュボード機能
│   └── api/               # APIエンドポイント
├── components/            # Reactコンポーネント
│   ├── ui/               # UIコンポーネント
│   └── features/         # 機能別コンポーネント
├── lib/                  # ユーティリティ関数
├── prisma/               # データベース設定
└── public/              # 静的ファイル
```

## 貢献について

1. このリポジトリをフォーク
2. 新しいブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add some amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。 