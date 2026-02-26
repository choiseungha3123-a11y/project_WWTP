"use client";

import { useState, useEffect, useCallback } from "react";
import Image from "next/image";
import { 
  CheckCircle2, Trash2, PencilLine, ImageIcon, 
  Loader2, User, Clock, X, Download, ListChecks, History
} from "lucide-react";

// --- 타입 정의 ---
interface MemoItem {
  memoNo: number;
  content: string;
  fileName: string | null;
  createTime: string;
  createMember: {
    userName: string;
  };
}

// 2. 백엔드에서 내려주는 "dataList": [ { items: [...] } ] 구조를 위한 타입
interface MemoListData {
  items: MemoItem[];
  totalCount?: number;
}

// 3. 공통 API 응답 (제네릭 T를 dataList의 '요소' 타입으로 사용)
interface ApiResponse<T> {
  success: boolean;
  dataList: T[]; // T 타입의 요소들이 담긴 배열
  errorMsg?: string;
}

// --- 이미지 컴포넌트 ---
function MemoImage({ memoNo, fileName }: { memoNo: number; fileName: string }) {
  const [imgUrl, setImgUrl] = useState<string | null>(null);
  const [showDetail, setShowDetail] = useState(false);

  useEffect(() => {
    const loadImg = async () => {
      try {
        const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/board/memo/image?memo_no=${memoNo}`, {
          headers: { "Authorization": `Bearer ${localStorage.getItem("accessToken")}` }
        });
        const result: ApiResponse<string | null> = await res.json();
        // dataList[1]은 contentType, dataList[2]는 base64 데이터
        if (result.success && result.dataList?.[2]) {
          setImgUrl(`data:${result.dataList[1]};base64,${result.dataList[2]}`);
        }
      } catch (e) {
        console.error("Image load error:", e);
      }
    };
    loadImg();
  }, [memoNo]);

  if (!imgUrl) return null;

  return (
    <>
      {/* 썸네일 영역 */}
      <div className="mt-3 relative group w-fit cursor-zoom-in" onClick={() => setShowDetail(true)}>
        <div className="relative w-32 h-24 overflow-hidden rounded-xl border border-white/10 shadow-lg">
          <Image 
            src={imgUrl} 
            alt={fileName} 
            fill 
            className="object-cover transition-transform duration-300 group-hover:scale-105"
            unoptimized // Base64는 별도 최적화가 필요 없으므로 unoptimized 적용
          />
        </div>
        <div className="absolute inset-0 bg-black/20 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center rounded-xl">
          <ImageIcon className="w-5 h-5 text-white" />
        </div>
      </div>

      {/* 원본 보기 모달 */}
      {showDetail && (
        <div className="fixed inset-0 z-120 flex items-center justify-center p-6 bg-slate-950/90 backdrop-blur-xl">
          <div className="absolute inset-0" onClick={() => setShowDetail(false)} />
          <div className="relative bg-slate-900 border border-white/20 rounded-[40px] overflow-hidden max-w-3xl w-full shadow-2xl animate-in zoom-in-95">
            <div className="flex items-center justify-between p-5 border-b border-white/10 bg-white/5">
              <span className="text-sm font-bold text-slate-200">{fileName}</span>
              <div className="flex gap-2">
                 <a href={imgUrl} download={fileName} className="p-2 text-slate-400 hover:text-white transition-colors">
                    <Download className="w-5 h-5" />
                 </a>
                 <button onClick={() => setShowDetail(false)} className="p-2 text-slate-400 hover:text-white">
                    <X className="w-5 h-5" />
                 </button>
              </div>
            </div>
            <div className="p-4 flex justify-center bg-black/20 relative min-h-75">
              <Image 
                src={imgUrl} 
                alt="원본" 
                width={800} 
                height={600} 
                className="max-h-[65vh] w-auto h-auto object-contain rounded-lg shadow-2xl"
                unoptimized
                priority // 모달 이미지는 우선순위 로드
              />
            </div>
          </div>
        </div>
      )}
    </>
  );
}

// --- 메인 패널 ---
export default function Row4ActionPanel() {
  const [listType, setListType] = useState<"pending" | "completed" | null>(null);
  const [memos, setMemos] = useState<MemoItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editContent, setEditContent] = useState("");

  const fetchMemos = useCallback(async (type: "pending" | "completed") => {
  setLoading(true);
  setMemos([]);
  try {
    const endpoint = type === "pending" ? "/api/board/memo/list" : "/api/board/memo/oldList";
    const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}${endpoint}?page=0&count=20`, {
      headers: { "Authorization": `Bearer ${localStorage.getItem("accessToken")}` }
    });

      const result: ApiResponse<MemoListData> = await res.json();
      
      if (result.success && result.dataList && result.dataList.length > 0) {
      // result.dataList[0]은 이제 MemoListData 타입이므로 .items 접근이 가능합니다.
      const listData = result.dataList[0];
      const fetchedItems = listData.items || [];
      
      // 원본 배열을 보존하며 정렬하기 위해 스프레드 연산자 사용을 권장합니다.
      const sortedItems = [...fetchedItems].sort((a, b) => 
        new Date(b.createTime).getTime() - new Date(a.createTime).getTime()
      );
      
      setMemos(sortedItems);
    }
  } catch (e) {
    console.error("Fetch Error:", e);
  } finally {
    setLoading(false);
  }
}, []);

  useEffect(() => {
    if (listType) fetchMemos(listType);
  }, [listType, fetchMemos]);

  useEffect(() => {
    const handleRefresh = () => {
      if (listType === "pending") fetchMemos("pending");
    };
    window.addEventListener("refreshMemoList", handleRefresh);
    return () => window.removeEventListener("refreshMemoList", handleRefresh);
  }, [listType, fetchMemos]);

  const handleAction = async (action: 'disable' | 'delete' | 'modify', id: number, content?: string) => {
    if (!confirm("진행하시겠습니까?")) return;
    
    const isModify = action === 'modify';
    const url = `${process.env.NEXT_PUBLIC_API_URL}/api/board/memo/${action}`;
    const token = localStorage.getItem("accessToken");
    
    let body: BodyInit;
    if (isModify) {
      const formData = new FormData();
      formData.append("memoNo", String(id));
      formData.append("content", content || "");
      body = formData;
    } else {
      body = JSON.stringify({ memoNo: id });
    }

    try {
      const res = await fetch(url, {
        method: "POST",
        headers: isModify 
          ? { "Authorization": `Bearer ${token}` } 
          : { "Content-Type": "application/json", "Authorization": `Bearer ${token}` },
        body
      });
      const result: ApiResponse<MemoListData> = await res.json();
      if (result.success) {
        setEditingId(null);
        if (listType) fetchMemos(listType);
      }
    } catch (e) { 
      console.error(e); 
    }
  };

  return (
    <div className="grid grid-cols-2 gap-4 h-full">
      <button 
        onClick={() => setListType("pending")} 
        className="bg-blue-600/10 border border-blue-500/20 rounded-4xl flex flex-col items-center justify-center gap-3 hover:bg-blue-600/20 transition-all group"
      >
        <div className="p-4 bg-blue-500/20 rounded-2xl group-hover:scale-110 transition-transform">
          <ListChecks className="w-10 h-10 text-blue-400" />
        </div>
        <span className="text-sm font-black text-blue-100">조치 예정 목록</span>
      </button>

      <button 
        onClick={() => setListType("completed")} 
        className="bg-emerald-600/10 border border-emerald-500/20 rounded-4xl flex flex-col items-center justify-center gap-3 hover:bg-emerald-600/20 transition-all group"
      >
        <div className="p-4 bg-emerald-500/20 rounded-2xl group-hover:scale-110 transition-transform">
          <History className="w-10 h-10 text-emerald-400" />
        </div>
        <span className="text-sm font-black text-emerald-100">조치 완료 기록</span>
      </button>

      {listType && (
        <div className="fixed inset-0 z-100 flex items-center justify-center p-6 bg-black/70 backdrop-blur-sm">
          <div className="bg-slate-900 border border-white/10 rounded-[48px] w-full max-w-4xl h-[85vh] flex flex-col shadow-2xl overflow-hidden animate-in slide-in-from-bottom-12 duration-300">
            {/* 헤더 */}
            <div className="p-8 border-b border-white/5 flex justify-between items-center bg-white/2">
              <div className="flex items-center gap-4">
                <div className={`p-3 rounded-2xl ${listType === 'pending' ? 'bg-blue-500/20 text-blue-400' : 'bg-emerald-500/20 text-emerald-400'}`}>
                  {listType === 'pending' ? <ListChecks className="w-6 h-6"/> : <History className="w-6 h-6"/>}
                </div>
                <h2 className="text-xl font-black text-white">{listType === 'pending' ? '조치 예정 목록' : '조치 완료 기록'}</h2>
              </div>
              <button onClick={() => setListType(null)} className="p-3 hover:bg-white/10 rounded-full text-slate-400">
                <X className="w-7 h-7"/>
              </button>
            </div>

            {/* 리스트 영역 */}
            <div className="flex-1 overflow-y-auto p-8 space-y-5 custom-scrollbar bg-slate-900/50">
              {loading ? (
                <div className="flex flex-col items-center justify-center h-full gap-4 text-slate-500">
                  <Loader2 className="w-10 h-10 animate-spin text-blue-500" />
                  <p>데이터를 불러오는 중입니다...</p>
                </div>
              ) : memos.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-full text-slate-700">
                  <p className="text-sm font-medium">표시할 데이터가 없습니다.</p>
                </div>
              ) : (
                memos.map((item) => (
                  <div key={item.memoNo} className="bg-white/3 border border-white/5 rounded-4xl p-6 group">
                    {editingId === item.memoNo ? (
                      <div className="space-y-4">
                        <textarea 
                          className="w-full bg-slate-950 border border-white/10 rounded-2xl p-4 text-sm text-white min-h-30" 
                          value={editContent} 
                          onChange={(e) => setEditContent(e.target.value)} 
                        />
                        <div className="flex justify-end gap-3">
                          <button onClick={() => setEditingId(null)} className="px-5 py-2 text-xs text-slate-500">취소</button>
                          <button 
                            onClick={() => handleAction('modify', item.memoNo, editContent)} 
                            className="px-8 py-2.5 bg-blue-600 text-white rounded-xl text-xs font-black"
                          >
                            저장
                          </button>
                        </div>
                      </div>
                    ) : (
                      <div className="flex justify-between items-start gap-8">
                        <div className="flex-1">
                          <p className="text-slate-200 text-base leading-relaxed whitespace-pre-wrap">{item.content}</p>
                          {item.fileName && <MemoImage memoNo={item.memoNo} fileName={item.fileName} />}
                          <div className="flex items-center gap-6 mt-6 pt-4 border-t border-white/5 text-xs text-slate-500 font-bold">
                            <span className="flex items-center gap-2"><User className="w-3.5 h-3.5"/> {item.createMember.userName}</span>
                            <span className="flex items-center gap-2"><Clock className="w-3.5 h-3.5"/> {new Date(item.createTime).toLocaleString()}</span>
                          </div>
                        </div>
                        <div className="flex flex-col gap-2 opacity-0 group-hover:opacity-100 transition-all">
                          {listType === "pending" && (
                            <>
                              <button onClick={() => handleAction('disable', item.memoNo)} className="p-3 bg-emerald-500/10 text-emerald-500 hover:bg-emerald-500 hover:text-white rounded-2xl transition-all">
                                <CheckCircle2 className="w-6 h-6" />
                              </button>
                              <button 
                                onClick={() => { setEditingId(item.memoNo); setEditContent(item.content); }} 
                                className="p-3 bg-blue-500/10 text-blue-400 hover:bg-blue-500 hover:text-white rounded-2xl transition-all"
                              >
                                <PencilLine className="w-6 h-6" />
                              </button>
                            </>
                          )}
                          <button onClick={() => handleAction('delete', item.memoNo)} className="p-3 bg-red-500/10 text-red-500 hover:bg-red-500 hover:text-white rounded-2xl transition-all">
                            <Trash2 className="w-6 h-6" />
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}