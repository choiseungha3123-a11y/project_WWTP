"use client";

import { useState, useEffect, useCallback } from "react";
import { Send, AlertCircle, CheckCircle2, Trash2, User, Edit2, X, Save } from "lucide-react";

interface Memo {
  memoNo: number;
  content: string;
  createTime: string;
  createMember: {
    userId: string;
    userName: string;
  };
}

export default function Row5ActionPanel() {
  const [memoInput, setMemoInput] = useState("");
  const [memos, setMemos] = useState<Memo[]>([]);
  const [loading, setLoading] = useState(false);

  // --- 수정 기능을 위한 상태 ---
  const [editingMemoNo, setEditingMemoNo] = useState<number | null>(null);
  const [editContent, setEditContent] = useState("");

  // 인증 헤더 생성 함수
  const getAuthHeaders = useCallback((): HeadersInit => {
    const token = localStorage.getItem("accessToken");
    if (!token) return { "Content-Type": "application/json" };
    
    const cleanToken = token.startsWith("Bearer ") ? token.replace("Bearer ", "") : token;
    return {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${cleanToken.trim()}`,
    };
  }, []);

  // API: 메모 목록 조회
  const fetchMemos = useCallback(async () => {
    try {
      const res = await fetch("/api/board/memo/list?page=0&count=10", {
        headers: getAuthHeaders(),
      });
      const result = await res.json();
      
      // 기존 코드의 데이터 구조(dataList[0].items)에 맞춰 수정
      if (result.success && result.dataList?.[0]?.items) {
        setMemos(result.dataList[0].items);
      }
    } catch (err) {
      console.error("메모 로드 실패:", err);
    }
  }, [getAuthHeaders]);

  useEffect(() => {
    fetchMemos();
  }, [fetchMemos]);

  // API: 메모 등록
  const handleRegister = async () => {
    if (!memoInput.trim()) return;
    setLoading(true);
    try {
      const res = await fetch("/api/board/memo/create", {
        method: "PUT",
        headers: getAuthHeaders(),
        body: JSON.stringify({ content: memoInput }),
      });
      if ((await res.json()).success) {
        setMemoInput("");
        fetchMemos();
      }
    } finally {
      setLoading(false);
    }
  };

  // API: 메모 수정 (추가됨)
  const handleUpdate = async (memoNo: number) => {
    if (!editContent.trim()) return;
    try {
      const res = await fetch("/api/board/memo/modify", {
        method: "POST",
        headers: getAuthHeaders(),
        body: JSON.stringify({ memoNo, content: editContent }),
      });
      if ((await res.json()).success) {
        setEditingMemoNo(null);
        fetchMemos();
      }
    } catch (err) {
      console.error("수정 실패:", err);
    }
  };

  // API: 조치 완료 (Disable)
  const handleComplete = async (memoNo: number) => {
    if (!confirm("이 조치 사항을 완료 처리하시겠습니까?")) return;
    try {
      const res = await fetch("/api/board/memo/disable", {
        method: "POST",
        headers: getAuthHeaders(),
        body: JSON.stringify({ memoNo }),
      });
      if ((await res.json()).success) fetchMemos();
    } catch (err) {
      console.error(err);
    }
  };

  // API: 삭제 (Delete)
  const handleDelete = async (memoNo: number) => {
    if (!confirm("기록을 영구 삭제하시겠습니까?")) return;
    try {
      const res = await fetch("/api/board/memo/delete", {
        method: "POST",
        headers: getAuthHeaders(),
        body: JSON.stringify({ memoNo }),
      });
      if ((await res.json()).success) fetchMemos();
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div className="bg-slate-800/50 p-5 rounded-3xl border border-white/10 h-full flex flex-col shadow-lg">
      <div className="shrink-0 mb-3">
        <h3 className="text-[15px] font-bold text-blue-400 flex items-center gap-2 mb-3">
          <AlertCircle className="w-4 h-4" /> 운영 권고 & 조치
        </h3>
        <div className="bg-blue-500/10 px-3 py-2 rounded-xl border border-blue-500/20 text-xs text-blue-100 flex items-center gap-2">
          <span className="text-blue-400">💡</span> 송풍량 +10% 권장 (질산화 저하 위험)
        </div>
      </div>

      <div className="shrink-0 flex gap-2 mb-4">
        <input 
          type="text"
          value={memoInput}
          onChange={(e) => setMemoInput(e.target.value)}
          placeholder="조치 사항 입력..."
          onKeyDown={(e) => e.key === 'Enter' && handleRegister()}
          className="flex-1 bg-slate-900/50 border border-white/10 rounded-xl px-4 py-2 text-xs text-slate-200 focus:border-blue-500 outline-none transition-all"
        />
        <button onClick={handleRegister} disabled={loading} className="bg-blue-600 hover:bg-blue-500 px-4 rounded-xl font-bold text-xs text-white flex items-center gap-1 shadow-lg disabled:opacity-50 transition-all">
          <Send className="w-3 h-3" /> 등록
        </button>
      </div>

      <div className="border-t border-white/5 mb-2 shrink-0"></div>

      <div className="flex-1 min-h-0 overflow-y-auto pr-1 custom-scrollbar">
        {memos.length === 0 ? (
          <div className="h-full flex items-center justify-center text-slate-600 text-xs">등록된 조치 이력이 없습니다.</div>
        ) : (
          <div className="space-y-2">
            {memos.map((memo) => (
              <div key={memo.memoNo} className="bg-slate-700/30 p-3 rounded-xl border border-white/5 hover:border-white/10 transition-colors group">
                {editingMemoNo === memo.memoNo ? (
                  // --- 수정 모드 UI ---
                  <div className="mb-2">
                    <textarea 
                      value={editContent}
                      onChange={(e) => setEditContent(e.target.value)}
                      className="w-full bg-slate-900 border border-blue-500/50 rounded-lg p-2 text-xs text-white outline-none"
                      rows={2}
                    />
                    <div className="flex gap-1 mt-2">
                      <button onClick={() => handleUpdate(memo.memoNo)} className="p-1.5 rounded-md bg-blue-600 text-white hover:bg-blue-500 transition-colors">
                        <Save className="w-3 h-3" />
                      </button>
                      <button onClick={() => setEditingMemoNo(null)} className="p-1.5 rounded-md bg-slate-600 text-white hover:bg-slate-500 transition-colors">
                        <X className="w-3 h-3" />
                      </button>
                    </div>
                  </div>
                ) : (
                  // --- 일반 모드 UI ---
                  <>
                    <p className="text-xs text-slate-200 leading-relaxed break-all mb-2">{memo.content}</p>
                    <div className="flex justify-between items-end">
                      <div className="flex flex-col gap-0.5">
                        <span className="text-[10px] text-slate-500 flex items-center gap-1"><User className="w-2.5 h-2.5" /> {memo.createMember?.userName}</span>
                        <span className="text-[9px] text-slate-600 font-mono">{new Date(memo.createTime).toLocaleDateString()}</span>
                      </div>
                      <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                        <button onClick={() => { setEditingMemoNo(memo.memoNo); setEditContent(memo.content); }} className="p-1.5 rounded-lg bg-white/5 hover:bg-white/10 text-slate-400"><Edit2 className="w-3 h-3" /></button>
                        <button onClick={() => handleComplete(memo.memoNo)} className="p-1.5 rounded-lg bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-500"><CheckCircle2 className="w-3 h-3" /></button>
                        <button onClick={() => handleDelete(memo.memoNo)} className="p-1.5 rounded-lg bg-red-500/10 hover:bg-red-500/20 text-red-400"><Trash2 className="w-3 h-3" /></button>
                      </div>
                    </div>
                  </>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}