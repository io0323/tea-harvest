import { auth } from '@clerk/nextjs';
import { redirect } from 'next/navigation';
import { Button } from '@/components/ui/button';
import Link from 'next/link';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Separator } from "@/components/ui/separator";

/**
 * 設定管理ページ
 * ユーザーがアカウントと予測の設定を管理するためのページ
 */
export default async function SettingsPage() {
  const { userId } = await auth();

  if (!userId) {
    redirect('/');
  }

  return (
    <div className="container mx-auto py-8 px-4">
      <div className="space-y-8">
        <div className="flex items-center justify-between">
          <h1 className="text-3xl font-bold">設定</h1>
          <Link href="/dashboard">
            <Button variant="outline">
              ダッシュボードに戻る
            </Button>
          </Link>
        </div>

        <div className="grid gap-6">
          {/* 通知設定 */}
          <Card>
            <CardHeader>
              <CardTitle>通知設定</CardTitle>
              <CardDescription>
                予測結果や重要なお知らせに関する通知設定を管理します
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="space-y-1">
                  <Label>予測完了通知</Label>
                  <p className="text-sm text-muted-foreground">
                    予測が完了したときにメール通知を受け取る
                  </p>
                </div>
                <Switch />
              </div>
              <div className="flex items-center justify-between">
                <div className="space-y-1">
                  <Label>週間サマリー</Label>
                  <p className="text-sm text-muted-foreground">
                    週に1回、予測結果のサマリーをメールで受け取る
                  </p>
                </div>
                <Switch />
              </div>
            </CardContent>
          </Card>

          {/* 予測設定 */}
          <Card>
            <CardHeader>
              <CardTitle>予測設定</CardTitle>
              <CardDescription>
                収穫時期予測のパラメータを設定します
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="space-y-1">
                  <Label>高精度モード</Label>
                  <p className="text-sm text-muted-foreground">
                    より多くのデータポイントを使用して精度を向上（処理時間が増加）
                  </p>
                </div>
                <Switch />
              </div>
              <div className="flex items-center justify-between">
                <div className="space-y-1">
                  <Label>自動データ更新</Label>
                  <p className="text-sm text-muted-foreground">
                    新しい気象データが利用可能になったら自動的に予測を更新
                  </p>
                </div>
                <Switch />
              </div>
            </CardContent>
          </Card>

          {/* データ管理 */}
          <Card>
            <CardHeader>
              <CardTitle>データ管理</CardTitle>
              <CardDescription>
                保存されているデータの管理と削除
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-4">
                <div>
                  <Label>保存データ</Label>
                  <p className="text-sm text-muted-foreground mt-1">
                    現在保存されている気象データと予測結果
                  </p>
                </div>
                <div className="flex gap-4">
                  <Button variant="outline">データをエクスポート</Button>
                  <Button variant="destructive">すべてのデータを削除</Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
} 