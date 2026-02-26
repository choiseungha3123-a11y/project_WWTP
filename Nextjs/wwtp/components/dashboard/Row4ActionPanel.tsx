"use client";

import { useState, useCallback } from "react";
import { ListChecks, History, MessageSquare, X, User, Clock, Trash2, CheckCircle2 } from "lucide-react";

export default function Row4ActionPanel() {
  const [activeModal, setActiveModal] = useState<"pending" | "completed" | "post" | null>(null);

  const renderModalContent = () => {
    switch(activeModal) {
      case "pending": return <MemoList type="pending" onClose={() => setActiveModal(null)} />;
      case "completed": return <MemoList type="completed" onClose={() => setActiveModal(null)} />;
      case "post": return <div className="p-20 text-center text-slate-500 font-bold">이용자 게시글 서비스 준비 중입니다.</div>;
      default: return null;
    }
  };

  return (
    // h-full 대신 최소 높이(h-24 등)를 지정하여 상단 공간을 확보합니다.
    <div className="p-3 h-20 flex flex-col justify-center bg-slate-900/40 rounded-3xl border border-white/5 shadow-inner">
      <div className="grid grid-cols-3 gap-3 h-full">
        <ActionButton 
          title="조치 예정 목록" 
          icon={<ListChecks className="w-5 h-4" />} 
          active={activeModal === "pending"}
          onClick={() => setActiveModal("pending")} 
        />
        <ActionButton 
          title="조치 완료 목록" 
          icon={<History className="w-5 h-4" />} 
          active={activeModal === "completed"}
          onClick={() => setActiveModal("completed")} 
        />
        <ActionButton 
          title="이용자 게시글" 
          icon={<MessageSquare className="w-5 h-4" />} 
          active={activeModal === "post"}
          onClick={() => setActiveModal("post")} 
        />
      </div>

      {/* 모달 로직은 동일 */}
      {activeModal && (
        <div className="fixed inset-0 z-100 flex items-center justify-center bg-black/75 backdrop-blur-md p-6">
          <div className="bg-slate-900 border border-white/10 w-full max-w-4xl max-h-[85vh] rounded-3xl overflow-hidden flex flex-col shadow-2xl">
            <div className="flex justify-between items-center p-5 border-b border-white/5 bg-slate-800/50">
              <h3 className="text-lg font-bold text-white flex items-center gap-2">
                {activeModal === "pending" ? "📋 조치 예정 목록" : activeModal === "completed" ? "✅ 조치 완료 내역" : "💬 이용자 게시글"}
              </h3>
              <button onClick={() => setActiveModal(null)} className="p-2 hover:bg-white/10 rounded-full text-slate-400">
                <X className="w-6 h-6" />
              </button>
            </div>
            <div className="flex-1 overflow-y-auto p-6 custom-scrollbar">
              {renderModalContent()}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

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

// --- MemoList 컴포넌트는 이전과 동일하므로 생략 ---

// --- 하위 컴포넌트: MemoList (중앙 팝업 내부 로직) ---
function MemoList({ type, onClose }: { type: "pending" | "completed", onClose: () => void }) {
  const [data, setData] = useState<any[]>([]);
  
  // 데이터 페칭 로직 (type에 따라 endpoint 변경 가능)
  const fetchItems = useCallback(async () => {
    const endpoint = type === "pending" ? "/api/board/memo/list" : "/api/board/memo/old-list"; 
    // 실제 API 구조에 맞게 수정 필요
    try {
      const res = await fetch(`${endpoint}?page=0&count=20`, {
        headers: { "Authorization": `Bearer ${localStorage.getItem("accessToken")}` }
      });
      const result = await res.json();
      if (result.success) setData(result.dataList?.[0]?.items || []);
    } catch (e) { console.error(e); }
  }, [type]);

  useState(() => { fetchItems(); });

  return (
    <div className="space-y-4">
      {data.length === 0 ? (
        <div className="py-20 text-center text-slate-600 italic">표시할 데이터가 없습니다.</div>
      ) : (
        data.map((item: any) => (
          <div key={item.memoNo} className="bg-white/5 p-4 rounded-2xl border border-white/5 flex justify-between items-center group">
            <div className="flex flex-col gap-1">
              <p className="text-slate-200 font-medium leading-relaxed">{item.content}</p>
              <div className="flex gap-4 mt-2">
                <span className="text-[11px] text-slate-500 flex items-center gap-1"><User className="w-3 h-3"/> {item.createMember.userName}</span>
                <span className="text-[11px] text-slate-500 flex items-center gap-1"><Clock className="w-3 h-3"/> {new Date(item.createTime).toLocaleString()}</span>
              </div>
            </div>
            <div className="flex gap-2 opacity-0 group-hover:opacity-100 transition-all">
              {type === "pending" && (
                <button className="p-2 bg-emerald-500/20 text-emerald-500 rounded-lg hover:bg-emerald-500 hover:text-white transition-colors">
                  <CheckCircle2 className="w-5 h-5" />
                </button>
              )}
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