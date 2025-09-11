'use client';

import { SignInButton, SignUpButton } from '@clerk/nextjs';
import { Button } from '@/components/ui/button';

export default function AuthButtons() {
  return (
    <div className="flex flex-col sm:flex-row gap-4 justify-center">
      <SignInButton mode="modal">
        <Button variant="default" size="lg">
          サインイン
        </Button>
      </SignInButton>
      <SignUpButton mode="modal">
        <Button variant="outline" size="lg">
          新規登録
        </Button>
      </SignUpButton>
    </div>
  );
} 