import { auth } from '@clerk/nextjs';
import { redirect } from 'next/navigation';
import { Button } from '@/components/ui/button';
import Link from 'next/link';

/**
 * 気象データアップロードページ
 * ユーザーが気象データをアップロードして収穫時期を予測するためのページ
 */
export default async function UploadPage() {
  const { userId } = await auth();

  if (!userId) {
    redirect('/');
  }

  return (
    <div className="container mx-auto py-8 px-4">
      <div className="space-y-8">
        {/* ヘッダー */}
        <div className="flex items-center justify-between">
          <h1 className="text-3xl font-bold">気象データアップロード</h1>
          <Link href="/dashboard">
            <Button variant="outline">
              ダッシュボードに戻る
            </Button>
          </Link>
        </div>

        {/* アップロードフォーム */}
        <div className="max-w-2xl mx-auto">
          <div className="p-6 bg-card rounded-lg border shadow-sm">
            <h2 className="text-xl font-semibold mb-4">データファイルの選択</h2>
            <p className="text-muted-foreground mb-6">
              CSVまたはExcelファイル形式の気象データをアップロードしてください。
            </p>
            
            {/* TODO: ファイルアップロードコンポーネントの実装 */}
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
              <p className="text-muted-foreground">
                ここにファイルをドラッグ＆ドロップ
                <br />
                または
              </p>
              <Button className="mt-4">
                ファイルを選択
              </Button>
            </div>

            {/* アップロードボタン */}
            <div className="mt-6">
              <Button className="w-full" disabled>
                アップロードして予測開始
              </Button>
            </div>
          </div>

          {/* 注意事項 */}
          <div className="mt-6 p-4 bg-muted rounded-lg">
            <h3 className="font-semibold mb-2">データ形式について</h3>
            <ul className="list-disc list-inside space-y-2 text-sm text-muted-foreground">
              <li>対応ファイル形式: CSV, Excel (.xlsx)</li>
              <li>必要なデータ: 気温、湿度、降水量、日照時間</li>
              <li>1行目はヘッダー行として扱われます</li>
              <li>データは日付順に並べてください</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
} 