import { auth } from '@clerk/nextjs';
import { redirect } from 'next/navigation';
import { Button } from '@/components/ui/button';
import Link from 'next/link';
import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { formatDate } from '@/lib/utils';

// モックデータ
const predictions = [
  {
    id: 1,
    region: "静岡",
    predictedDate: "2024-05-15",
    confidence: 0.95,
    createdAt: "2024-03-01",
    status: "高精度",
    details: {
      temperature: "平均22℃",
      rainfall: "適度",
      humidity: "65%",
      notes: "理想的な気象条件が続いています。",
    },
  },
  {
    id: 2,
    region: "京都",
    predictedDate: "2024-05-20",
    confidence: 0.85,
    createdAt: "2024-03-02",
    status: "中精度",
    details: {
      temperature: "平均20℃",
      rainfall: "やや多め",
      humidity: "70%",
      notes: "降水量が平年より多めですが、収穫への影響は限定的です。",
    },
  },
  {
    id: 3,
    region: "鹿児島",
    predictedDate: "2024-05-10",
    confidence: 0.75,
    createdAt: "2024-03-03",
    status: "要注意",
    details: {
      temperature: "平均24℃",
      rainfall: "少なめ",
      humidity: "60%",
      notes: "乾燥傾向が続いているため、灌水管理に注意が必要です。",
    },
  },
  {
    id: 4,
    region: "宮崎",
    predictedDate: "2024-05-18",
    confidence: 0.92,
    createdAt: "2024-03-04",
    status: "高精度",
    details: {
      temperature: "平均23℃",
      rainfall: "適度",
      humidity: "68%",
      notes: "生育に最適な気象条件が整っています。",
    },
  },
  {
    id: 5,
    region: "三重",
    predictedDate: "2024-05-25",
    confidence: 0.88,
    createdAt: "2024-03-05",
    status: "中精度",
    details: {
      temperature: "平均21℃",
      rainfall: "やや少なめ",
      humidity: "63%",
      notes: "気温は安定していますが、降水量の確保が必要です。",
    },
  },
  {
    id: 6,
    region: "熊本",
    predictedDate: "2024-05-12",
    confidence: 0.94,
    createdAt: "2024-03-06",
    status: "高精度",
    details: {
      temperature: "平均22℃",
      rainfall: "適度",
      humidity: "67%",
      notes: "バランスの取れた気象条件で、高品質な収穫が期待できます。",
    },
  },
  {
    id: 7,
    region: "埼玉",
    predictedDate: "2024-05-28",
    confidence: 0.78,
    createdAt: "2024-03-07",
    status: "要注意",
    details: {
      temperature: "平均19℃",
      rainfall: "多め",
      humidity: "75%",
      notes: "湿度が高めで、病害の発生に注意が必要です。",
    },
  },
  {
    id: 8,
    region: "愛知",
    predictedDate: "2024-05-22",
    confidence: 0.91,
    createdAt: "2024-03-08",
    status: "高精度",
    details: {
      temperature: "平均21℃",
      rainfall: "適度",
      humidity: "66%",
      notes: "安定した気象条件が続いており、順調な生育が見込まれます。",
    },
  },
  {
    id: 9,
    region: "福岡",
    predictedDate: "2024-05-16",
    confidence: 0.86,
    createdAt: "2024-03-09",
    status: "中精度",
    details: {
      temperature: "平均20℃",
      rainfall: "やや多め",
      humidity: "69%",
      notes: "日照時間がやや少なめですが、生育への影響は軽微です。",
    },
  },
  {
    id: 10,
    region: "大分",
    predictedDate: "2024-05-14",
    confidence: 0.93,
    createdAt: "2024-03-10",
    status: "高精度",
    details: {
      temperature: "平均22℃",
      rainfall: "適度",
      humidity: "64%",
      notes: "理想的な気象条件が続いており、高品質な茶葉が期待できます。",
    },
  },
];

function getStatusBadge(status: string) {
  switch (status) {
    case "高精度":
      return <Badge className="bg-green-500">高精度</Badge>;
    case "中精度":
      return <Badge className="bg-yellow-500">中精度</Badge>;
    case "要注意":
      return <Badge className="bg-red-500">要注意</Badge>;
    default:
      return <Badge>不明</Badge>;
  }
}

function PredictionDetails({ prediction }: { prediction: typeof predictions[0] }) {
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-4">
        <div>
          <h4 className="font-semibold">気温</h4>
          <p className="text-sm text-muted-foreground">{prediction.details.temperature}</p>
        </div>
        <div>
          <h4 className="font-semibold">降水量</h4>
          <p className="text-sm text-muted-foreground">{prediction.details.rainfall}</p>
        </div>
        <div>
          <h4 className="font-semibold">湿度</h4>
          <p className="text-sm text-muted-foreground">{prediction.details.humidity}</p>
        </div>
        <div>
          <h4 className="font-semibold">信頼度</h4>
          <p className="text-sm text-muted-foreground">{(prediction.confidence * 100).toFixed(1)}%</p>
        </div>
      </div>
      <div>
        <h4 className="font-semibold">備考</h4>
        <p className="text-sm text-muted-foreground">{prediction.details.notes}</p>
      </div>
    </div>
  );
}

/**
 * 予測履歴表示ページ
 * ユーザーが過去の予測結果を確認するためのページ
 */
export default async function HistoryPage({
  searchParams,
}: {
  searchParams: { page?: string };
}) {
  const { userId } = await auth();

  if (!userId) {
    redirect('/');
  }

  // ページネーションの設定
  const page = Number(searchParams.page) || 1;
  const itemsPerPage = 3;
  const start = (page - 1) * itemsPerPage;
  const end = start + itemsPerPage;
  const totalPages = Math.ceil(predictions.length / itemsPerPage);
  const currentPredictions = predictions.slice(start, end);

  return (
    <div className="container mx-auto py-8 px-4">
      <div className="space-y-8">
        <div className="flex items-center justify-between">
          <h1 className="text-3xl font-bold">予測履歴</h1>
          <Link href="/dashboard">
            <Button variant="outline">
              ダッシュボードに戻る
            </Button>
          </Link>
        </div>

        <div className="space-y-4">
          {currentPredictions.map((prediction) => (
            <div key={prediction.id} className="p-6 bg-card rounded-lg border shadow-sm">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-xl font-semibold">予測 #{prediction.id}</h2>
                  <p className="text-muted-foreground">{prediction.createdAt}</p>
                </div>
                <Link href={`/dashboard/history/${prediction.id}`}>
                  <Button variant="outline">詳細を表示</Button>
                </Link>
              </div>
              <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4">
                <div>
                  <p className="text-sm text-muted-foreground">地域</p>
                  <p className="font-medium">{prediction.region}</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">予測収穫日</p>
                  <p className="font-medium">{prediction.predictedDate}</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">予測精度</p>
                  <p className="font-medium">{(prediction.confidence * 100).toFixed(1)}%</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">ステータス</p>
                  <p className="font-medium">{prediction.status}</p>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* ページネーション */}
        <div className="flex justify-center gap-2">
          <Link href={`/dashboard/history?page=${Math.max(1, page - 1)}`}>
            <Button variant="outline" disabled={page === 1}>前へ</Button>
          </Link>
          {Array.from({ length: totalPages }, (_, i) => i + 1).map((pageNum) => (
            <Link key={pageNum} href={`/dashboard/history?page=${pageNum}`}>
              <Button 
                variant="outline" 
                className={pageNum === page ? "bg-primary text-primary-foreground" : ""}
              >
                {pageNum}
              </Button>
            </Link>
          ))}
          <Link href={`/dashboard/history?page=${Math.min(totalPages, page + 1)}`}>
            <Button variant="outline" disabled={page === totalPages}>次へ</Button>
          </Link>
        </div>
      </div>
    </div>
  );
} 