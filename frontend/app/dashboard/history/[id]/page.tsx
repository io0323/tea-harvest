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

/**
 * 予測履歴詳細ページ
 * 特定の予測結果の詳細情報を表示する
 */
export default async function PredictionDetailPage({
  params,
}: {
  params: { id: string };
}) {
  const { userId } = await auth();

  if (!userId) {
    redirect('/');
  }

  // TODO: 実際のデータベースから予測データを取得
  const prediction = {
    id: params.id,
    region: "静岡",
    predictedDate: "2024-05-15",
    confidence: 0.95,
    createdAt: "2024-03-01",
    status: "高精度",
    details: {
      temperature: "平均22℃",
      rainfall: "適度",
      humidity: "65%",
      sunlight: "6.5時間/日",
      notes: "理想的な気象条件が続いています。",
    },
    metrics: {
      temperatureRange: "18℃ ~ 26℃",
      rainfallTotal: "150mm",
      humidityRange: "60% ~ 70%",
      sunlightTotal: "195時間",
    },
  };

  return (
    <div className="container mx-auto py-8 px-4">
      <div className="space-y-8">
        {/* ヘッダー */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold">予測 #{prediction.id}</h1>
            <p className="text-muted-foreground">
              {prediction.createdAt} 作成
            </p>
          </div>
          <Link href="/dashboard/history">
            <Button variant="outline">
              履歴一覧に戻る
            </Button>
          </Link>
        </div>

        <div className="grid gap-6">
          {/* 基本情報 */}
          <Card>
            <CardHeader>
              <CardTitle>基本情報</CardTitle>
              <CardDescription>予測の基本的な情報</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                <div>
                  <h3 className="font-semibold">地域</h3>
                  <p className="text-muted-foreground">{prediction.region}</p>
                </div>
                <div>
                  <h3 className="font-semibold">予測収穫日</h3>
                  <p className="text-muted-foreground">{prediction.predictedDate}</p>
                </div>
                <div>
                  <h3 className="font-semibold">予測精度</h3>
                  <p className="text-muted-foreground">{(prediction.confidence * 100).toFixed(1)}%</p>
                </div>
                <div>
                  <h3 className="font-semibold">ステータス</h3>
                  <p className="text-muted-foreground">{prediction.status}</p>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* 気象条件 */}
          <Card>
            <CardHeader>
              <CardTitle>気象条件</CardTitle>
              <CardDescription>予測期間中の気象条件の詳細</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                <div>
                  <h3 className="font-semibold">気温</h3>
                  <p className="text-muted-foreground">{prediction.details.temperature}</p>
                  <p className="text-sm text-muted-foreground">{prediction.metrics.temperatureRange}</p>
                </div>
                <div>
                  <h3 className="font-semibold">降水量</h3>
                  <p className="text-muted-foreground">{prediction.details.rainfall}</p>
                  <p className="text-sm text-muted-foreground">{prediction.metrics.rainfallTotal}</p>
                </div>
                <div>
                  <h3 className="font-semibold">湿度</h3>
                  <p className="text-muted-foreground">{prediction.details.humidity}</p>
                  <p className="text-sm text-muted-foreground">{prediction.metrics.humidityRange}</p>
                </div>
                <div>
                  <h3 className="font-semibold">日照時間</h3>
                  <p className="text-muted-foreground">{prediction.details.sunlight}</p>
                  <p className="text-sm text-muted-foreground">{prediction.metrics.sunlightTotal}</p>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* 備考 */}
          <Card>
            <CardHeader>
              <CardTitle>備考</CardTitle>
              <CardDescription>予測に関する追加情報</CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-muted-foreground">{prediction.details.notes}</p>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
} 