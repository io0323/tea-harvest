import { auth } from '@clerk/nextjs';
import { redirect } from 'next/navigation';
import { Button } from '@/components/ui/button';
import Link from 'next/link';

export default async function DashboardPage() {
  const { userId } = await auth();

  if (!userId) {
    redirect('/');
  }

  return (
    <div className="container mx-auto py-8 px-4">
      <div className="space-y-8">
        <div className="flex items-center justify-between">
          <h1 className="text-3xl font-bold">ダッシュボード</h1>
          <Link href="/">
            <Button variant="outline">
              ホームに戻る
            </Button>
          </Link>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {/* 気象データアップロード */}
          <div className="p-6 bg-card rounded-lg border shadow-sm">
            <h2 className="text-xl font-semibold mb-4">気象データ</h2>
            <p className="text-muted-foreground mb-4">
              気象データをアップロードして収穫時期を予測
            </p>
            <Button variant="outline" className="w-full" asChild>
              <Link href="/dashboard/upload">
                データをアップロード
              </Link>
            </Button>
          </div>

          {/* 予測履歴 */}
          <div className="p-6 bg-card rounded-lg border shadow-sm">
            <h2 className="text-xl font-semibold mb-4">予測履歴</h2>
            <p className="text-muted-foreground mb-4">
              過去の予測結果を確認
            </p>
            <Link href="/dashboard/history" passHref>
              <Button variant="outline" className="w-full">
                履歴を表示
              </Button>
            </Link>
          </div>

          {/* 設定 */}
          <div className="p-6 bg-card rounded-lg border shadow-sm">
            <h2 className="text-xl font-semibold mb-4">設定</h2>
            <p className="text-muted-foreground mb-4">
              アカウントと予測の設定を管理
            </p>
            <Button variant="outline" className="w-full" asChild>
              <Link href="/dashboard/settings">
                設定を開く
              </Link>
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
} 