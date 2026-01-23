'use client';

import { useState } from 'react';
import { authService } from '../services/authService';

export default function LoginPage() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);

    try {
      const data = await authService.login(email, password);
      console.log('로그인 성공:', data);
      
      // 성공 시 처리 (예: 메인 페이지로 이동)
      alert('로그인에 성공했습니다!');
      window.location.href = '/'; 
    } catch (error: any) {
      alert(error.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-50">
      <form 
        onSubmit={handleLogin} 
        className="w-full max-w-md p-8 bg-white shadow-lg rounded-xl border border-gray-200"
      >
        <h2 className="text-3xl font-extrabold text-center text-gray-800 mb-8">관리자 로그인</h2>
        
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">이메일</label>
            <input 
              type="email" 
              required
              className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 outline-none"
              placeholder="admin@example.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">비밀번호</label>
            <input 
              type="password" 
              required
              className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 outline-none"
              placeholder="••••••••"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </div>

          <button 
            type="submit" 
            disabled={isLoading}
            className={`w-full p-3 mt-6 text-white font-bold rounded-md transition-colors ${
              isLoading ? 'bg-gray-400' : 'bg-blue-600 hover:bg-blue-700'
            }`}
          >
            {isLoading ? '로그인 중...' : '로그인'}
          </button>
        </div>
      </form>
    </div>
  );
}