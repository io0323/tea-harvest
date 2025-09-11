import { auth } from '@clerk/nextjs';
import { redirect } from 'next/navigation';
import AuthButtons from '@/components/auth/AuthButtons';

export default async function Home() {
  const { userId } = await auth();

  if (userId) {
    redirect('/dashboard');
  }

  return (
    <div className="flex min-h-screen flex-col items-center justify-center p-6 md:p-24">
      <div className="w-full max-w-4xl space-y-8">
        <div className="text-center space-y-4">
          <h1 className="text-4xl font-bold">Tea Harvest</h1>
          <p className="text-lg text-muted-foreground">
            お茶の収穫を効率的に管理するアプリケーション
          </p>
        </div>
        <AuthButtons />
      </div>
    </div>
  );
} 