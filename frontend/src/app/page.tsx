import { Button } from '@/components/common/buttons/Button';
import { SignedIn, SignedOut, SignInButton, SignUpButton } from '@clerk/nextjs';
import Link from 'next/link';

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-24">
      <h1 className="text-4xl font-bold mb-8">Tea Harvest</h1>
      <p className="text-lg text-gray-600 mb-8">
        お茶の収穫を効率的に管理するアプリケーション
      </p>
      
      <SignedOut>
        <div className="flex gap-4">
          <SignInButton mode="modal">
            <Button variant="primary">
              サインイン
            </Button>
          </SignInButton>
          <SignUpButton mode="modal">
            <Button variant="outline">
              新規登録
            </Button>
          </SignUpButton>
        </div>
      </SignedOut>
      
      <SignedIn>
        <Link href="/dashboard">
          <Button variant="primary">
            ダッシュボードへ
          </Button>
        </Link>
      </SignedIn>
    </main>
  );
}
