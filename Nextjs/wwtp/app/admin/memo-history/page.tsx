"use client";

import { useState, useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";
import { ArrowLeft, Clock, FileText } from "lucide-react";

interface Memo {
  memoNo: number;
  content: string;
  createTime: string;
  createMember: {
    userId: string;
    userName: string;
  };
}

export default function MemoHistoryPage() {
  const router = useRouter();
  const [oldMemos, setOldMemos] = useState<Memo[]>([]);
  const [loading, setLoading] = useState(true);

  const getAuthHeaders = useCallback((): HeadersInit => {
    const token = localStorage.getItem("accessToken");
    const cleanToken = token?.startsWith("Bearer ") ? token.replace("Bearer ", "") : token;
    return {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${cleanToken?.trim()}`,
    };
  }, []);

  const fetchOldMemos = useCallback(async () => {
    try {
      const res = await fetch("/api/board/memo/OldList?page=0&count=50", {
        headers: getAuthHeaders(),
      });
      const result = await res.json();
      if (result.success && result.dataList?.[0]?.items) {
        const sortedItems = [...result.dataList[0].items].sort((a, b) => {
        return new Date(b.createTime).getTime() - new Date(a.createTime).getTime();
      });
      setOldMemos(sortedItems);
      }
    } catch (err) {
      console.error("이력 로드 실패:", err);
    } finally {
      setLoading(false);
    }
  }, [getAuthHeaders]);

  useEffect(() => {
    const role = localStorage.getItem("userRole");
    if (role !== "ROLE_ADMIN") {
      alert("권한이 없습니다.");
      router.push("/dashboard");
      return;
    }
    fetchOldMemos();
  }, [fetchOldMemos, router]);

  return (
    <div className="min-h-screen bg-[#0a0f1d] text-slate-200 p-8">
      <header className="max-w-6xl mx-auto mb-10 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <button 
            onClick={() => router.back()}
            className="p-2 hover:bg-white/5 rounded-full transition-colors"
          >
            <ArrowLeft className="w-6 h-6 text-slate-400" />
          </button>
          <div>
            <h1 className="text-2xl font-black text-white tracking-tight">조치 이력 기록 시트</h1>
            <p className="text-slate-500 text-xs uppercase tracking-widest mt-1">Completed Action History</p>
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto">
        <div className="bg-slate-900/50 border border-white/5 rounded-3xl overflow-hidden backdrop-blur-md">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="bg-white/5 text-slate-400 text-[10px] uppercase tracking-widest">
                <th className="px-6 py-4 font-bold">No.</th>
                <th className="px-6 py-4 font-bold">완료된 조치 내용</th>
                <th className="px-6 py-4 font-bold">작성자</th>
                <th className="px-6 py-4 font-bold">기록 시간</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {loading ? (
                <tr><td colSpan={4} className="text-center py-20 text-slate-500">데이터를 불러오는 중...</td></tr>
              ) : oldMemos.length === 0 ? (
                <tr><td colSpan={4} className="text-center py-20 text-slate-500">완료된 이력이 없습니다.</td></tr>
              ) : (
                oldMemos.map((memo) => (
                  <tr key={memo.memoNo} className="hover:bg-white/2 transition-colors group">
                    <td className="px-6 py-5 text-xs font-mono text-slate-500">#{memo.memoNo}</td>
                    <td className="px-6 py-5">
                      <div className="flex items-start gap-3">
                        <FileText className="w-4 h-4 text-blue-500 shrink-0 mt-0.5" />
                        <p className="text-sm text-slate-200 leading-relaxed">{memo.content}</p>
                      </div>
                    </td>
                    <td className="px-6 py-5 text-sm text-slate-300">
                      <div className="flex items-center gap-2">
                        <div className="w-6 h-6 rounded-full bg-slate-700 flex items-center justify-center text-[10px] font-bold">
                          {memo.createMember.userName.substring(0, 1)}
                        </div>
                        {memo.createMember.userName}
                      </div>
                    </td>
                    <td className="px-6 py-5 text-xs text-slate-500 font-mono">
                      <div className="flex items-center gap-2">
                        <Clock className="w-3 h-3" />
                        {new Date(memo.createTime).toLocaleString()}
                      </div>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </main>
    </div>
  );
}