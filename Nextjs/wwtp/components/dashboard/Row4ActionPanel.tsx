"use client";

import { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { 
  ListChecks, 
  History, 
  MessageSquare, 
  X, 
  User, 
  Clock, 
  Trash2, 
  CheckCircle2, 
  FileText, 
  Loader2 
} from "lucide-react";

// --- 인터페이스 정의 ---
interface Memo {
  memoNo: number;
  content: string;
  createTime: string;
  createMember: {
    userId: string;
    userName: string;
  };
}

export default function Row4ActionPanel() {
  const [activeModal, setActiveModal] = useState<"pending" | "completed" | "post" | null>(null);

  return (
    <div className="p-3 h-full flex flex-col justify-center">
      {/* 대시보드 하단 버튼 3개 */}
      <div className="grid grid-cols-3 gap-3 h-full">
        <ActionButton 
          title="조치 예정 목록" 
          icon={<ListChecks className="w-5 h-5" />} 
          active={activeModal === "pending"}
          onClick={() => setActiveModal("pending")} 
        />
        <ActionButton 
          title="조치 완료 이력" 
          icon={<History className="w-5 h-5" />} 
          active={activeModal === "completed"}
          onClick={() => setActiveModal("completed")} 
        />
        <ActionButton 
          title="이용자 게시글" 
          icon={<MessageSquare className="w-5 h-5" />} 
          active={activeModal === "post"}
          onClick={() => setActiveModal("post")} 
        />
      </div>

      {/* 중앙 모달 레이어 */}
      <AnimatePresence>
        {activeModal && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[100] flex items-center justify-center bg-black/80 backdrop-blur-md p-4 lg:p-10"
          >
            <motion.div 
              initial={{ scale: 0.95, opacity: 0, y: 20 }}
              animate={{ scale: 1, opacity: 1, y: 0 }}
              exit={{ scale: 0.95, opacity: 0, y: 20 }}
              className="bg-[#0a0f1d] border border-white/10 w-full max-w-5xl max-h-[85vh] rounded-[2.5rem] overflow-hidden flex flex-col shadow-2xl"
            >
              {/* 모달 헤더 */}
              <div className="flex justify-between items-center p-6 border-b border-white/5 bg-slate-800/50">
                <div>
                  <h3 className="text-xl font-bold text-white flex items-center gap-3">
                    {activeModal === "pending" && "📋 조치 예정 목록"}
                    {activeModal === "completed" && "📜 조치 완료 기록 시트"}
                    {activeModal === "post" && "💬 이용자 게시글"}
                  </h3>
                </div>
                <button 
                  onClick={() => setActiveModal(null)} 
                  className="p-3 hover:bg-white/10 rounded-full text-slate-400 transition-colors"
                >
                  <X className="w-6 h-6" />
                </button>
              </div>

              {/* 모달 본문 컨텐츠 */}
              <div className="flex-1 overflow-y-auto custom-scrollbar p-6 bg-[#0a0f1d]">
                {activeModal === "pending" && (
                  <PendingMemoList />
                )}
                
                {activeModal === "completed" && (
                  <CompletedHistoryTable />
                )}

                {activeModal === "post" && (
                  <div className="py-20 text-center text-slate-500 font-bold">
                    이용자 게시글 서비스 준비 중입니다.
                  </div>
                )}
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// --- 1. [조치 예정 목록] 컴포넌트 (기존 로직) ---
function PendingMemoList() {
  const [data, setData] = useState<Memo[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchItems = useCallback(async () => {
    try {
      const token = localStorage.getItem("accessToken");
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/board/memo/list?page=0&count=20`, {
        headers: { "Authorization": `Bearer ${token?.replace("Bearer ", "")}` }
      });
      const result = await res.json();
      if (result.success) setData(result.dataList?.[0]?.items || []);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchItems(); }, [fetchItems]);

  if (loading) return <div className="text-center py-20 text-slate-500">불러오는 중...</div>;

  return (
    <div className="space-y-4">
      {data.length === 0 ? (
        <div className="py-20 text-center text-slate-600 italic">표시할 데이터가 없습니다.</div>
      ) : (
        data.map((item) => (
          <div key={item.memoNo} className="bg-white/5 p-5 rounded-2xl border border-white/5 flex justify-between items-center group hover:bg-white/10 transition-all">
            <div className="flex flex-col gap-1">
              <p className="text-slate-200 font-medium leading-relaxed">{item.content}</p>
              <div className="flex gap-4 mt-2">
                <span className="text-[11px] text-slate-500 flex items-center gap-1">
                  <User className="w-3 h-3"/> {item.createMember.userName}
                </span>
                <span className="text-[11px] text-slate-500 flex items-center gap-1">
                  <Clock className="w-3 h-3"/> {new Date(item.createTime).toLocaleString()}
                </span>
              </div>
            </div>
            <div className="flex gap-2 opacity-0 group-hover:opacity-100 transition-all">
              <button className="p-2 bg-emerald-500/20 text-emerald-500 rounded-lg hover:bg-emerald-500 hover:text-white transition-colors">
                <CheckCircle2 className="w-5 h-5" />
              </button>
              <button className="p-2 bg-red-500/20 text-red-500 rounded-lg hover:bg-red-500 hover:text-white transition-colors">
                <Trash2 className="w-5 h-5" />
              </button>
            </div>
          </div>
        ))
      )}
    </div>
  );
}

// --- 2. [조치 완료 이력] 컴포넌트 (테이블 형태) ---
function CompletedHistoryTable() {
  const [memos, setMemos] = useState<Memo[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchMemos = useCallback(async () => {
    try {
      const token = localStorage.getItem("accessToken");
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/board/memo/oldList?page=0&count=50`, {
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${token?.replace("Bearer ", "")}`,
        },
      });
      const result = await res.json();
      if (result.success && result.dataList?.[0]?.items) {
        const sorted = [...result.dataList[0].items].sort((a, b) => 
          new Date(b.createTime).getTime() - new Date(a.createTime).getTime()
        );
        setMemos(sorted);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchMemos(); }, [fetchMemos]);

  if (loading) return (
    <div className="flex flex-col items-center justify-center py-32 gap-4">
      <Loader2 className="w-8 h-8 text-blue-500 animate-spin" />
      <p className="text-slate-500 font-medium">기록 데이터를 불러오는 중...</p>
    </div>
  );

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left border-separate border-spacing-y-2">
        <thead>
          <tr className="text-slate-500 text-[10px] uppercase tracking-widest px-6">
            <th className="px-6 py-3 font-bold">No.</th>
            <th className="px-6 py-3 font-bold">완료된 조치 내용</th>
            <th className="px-6 py-3 font-bold">작성자</th>
            <th className="px-6 py-3 font-bold text-right">기록 시간</th>
          </tr>
        </thead>
        <tbody>
          {memos.map((memo) => (
            <tr key={memo.memoNo} className="bg-white/2 hover:bg-white/5 transition-all group">
              <td className="px-6 py-4 first:rounded-l-2xl text-xs font-mono text-slate-500">#{memo.memoNo}</td>
              <td className="px-6 py-4">
                <div className="flex items-start gap-3">
                  <FileText className="w-4 h-4 text-blue-500 shrink-0 mt-0.5" />
                  <p className="text-sm text-slate-200 leading-relaxed">{memo.content}</p>
                </div>
              </td>
              <td className="px-6 py-4 text-sm text-slate-300">
                <div className="flex items-center gap-2">
                  <div className="w-6 h-6 rounded-full bg-blue-600/20 text-blue-400 flex items-center justify-center text-[10px] font-bold">
                    {memo.createMember.userName.substring(0, 1)}
                  </div>
                  {memo.createMember.userName}
                </div>
              </td>
              <td className="px-6 py-4 last:rounded-r-2xl text-xs text-slate-500 font-mono text-right">
                {new Date(memo.createTime).toLocaleString()}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// --- 하단 액션 버튼 컴포넌트 ---
function ActionButton({ title, icon, onClick, active }: { title: string, icon: React.ReactNode, onClick: () => void, active: boolean }) {
  return (
    <button 
      onClick={onClick}
      className={`flex flex-col items-center justify-center gap-2 rounded-2xl transition-all border
      ${active ? 'bg-blue-600/20 border-blue-500/50 text-blue-400 shadow-lg' 
               : 'bg-slate-800/60 border-white/5 text-slate-400 hover:bg-slate-700 hover:border-white/10'}`}
    >
      <div className={`${active ? 'text-blue-400' : 'text-slate-500'}`}>{icon}</div>
      <span className="font-bold text-[12px] tracking-tight">{title}</span>
    </button>
  );
}