/**
 * ダッシュボードページ
 * 認証済みユーザーのみアクセス可能
 */
import { auth } from "@clerk/nextjs";
import { redirect } from "next/navigation";

export default async function DashboardPage() {
  const { userId } = await auth();

  if (!userId) {
    redirect("/sign-in");
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-8">ダッシュボード</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {/* 収穫データの概要 */}
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <h2 className="text-xl font-semibold mb-4">収穫データ</h2>
          <p className="text-gray-600">まだデータがありません</p>
        </div>

        {/* 天気情報 */}
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <h2 className="text-xl font-semibold mb-4">天気情報</h2>
          <p className="text-gray-600">データ取得中...</p>
        </div>

        {/* タスク */}
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <h2 className="text-xl font-semibold mb-4">タスク</h2>
          <p className="text-gray-600">タスクはありません</p>
        </div>
      </div>
    </div>
  );
} 