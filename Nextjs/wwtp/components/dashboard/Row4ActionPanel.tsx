"use client";

import { useState, useEffect, useCallback } from "react";
import Image from "next/image";
import { 
  CheckCircle2, Trash2, PencilLine, ImageIcon, 
  Loader2, User, Clock, X, Download, ListChecks, History,
  MessageSquareQuote, MapPin
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

// [추가] 시민 게시글 타입 정의 (백엔드 PublicDTO 대응)
interface PublicItem {
  no: number;
  userNo: number;
  pos: string;
  content: string;
  picture: string | null; // byte[] 데이터
}

interface MemoListData {
  items: MemoItem[];
  totalCount?: number;
}

interface ApiResponse<T> {
  success: boolean;
  dataList: T[];
  errorMsg?: string;
}

interface Row4Props {
  isDarkMode?: boolean;
}

// --- 이미지 컴포넌트 (기존 유지) ---
function MemoImage({ memoNo, fileName, isDarkMode }: { memoNo: number; fileName: string; isDarkMode: boolean }) {
  const [imgUrl, setImgUrl] = useState<string | null>(null);
  const [showDetail, setShowDetail] = useState(false);

  useEffect(() => {
    const loadImg = async () => {
      try {
        const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/board/memo/image?memo_no=${memoNo}`, {
          headers: { "Authorization": `Bearer ${localStorage.getItem("accessToken")}` }
        });
        const result: ApiResponse<string | null> = await res.json();
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
      <div className="mt-3 relative group w-fit cursor-zoom-in" onClick={() => setShowDetail(true)}>
        <div className={`relative w-32 h-24 overflow-hidden rounded-xl border shadow-lg ${isDarkMode ? 'border-white/10' : 'border-slate-200'}`}>
          <Image 
            src={imgUrl} 
            alt={fileName} 
            fill 
            className="object-cover transition-transform duration-300 group-hover:scale-105"
            unoptimized 
          />
        </div>
        <div className="absolute inset-0 bg-black/20 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center rounded-xl">
          <ImageIcon className="w-5 h-5 text-white" />
        </div>
      </div>

      {showDetail && (
        <div className="fixed inset-0 z-120 flex items-center justify-center p-6 bg-slate-950/90 backdrop-blur-xl">
          <div className="absolute inset-0" onClick={() => setShowDetail(false)} />
          <div className={`relative border rounded-[40px] overflow-hidden max-w-3xl w-full shadow-2xl animate-in zoom-in-95 ${isDarkMode ? 'bg-slate-900 border-white/20' : 'bg-white border-slate-200'}`}>
            <div className={`flex items-center justify-between p-5 border-b ${isDarkMode ? 'border-white/10 bg-white/5' : 'border-slate-100 bg-slate-50'}`}>
              <span className={`text-sm font-bold ${isDarkMode ? 'text-slate-200' : 'text-slate-800'}`}>{fileName}</span>
              <div className="flex gap-2">
                 <a href={imgUrl} download={fileName} className="p-2 text-slate-400 hover:text-blue-500 transition-colors">
                    <Download className="w-5 h-5" />
                 </a>
                 <button onClick={() => setShowDetail(false)} className="p-2 text-slate-400 hover:text-red-500">
                    <X className="w-5 h-5" />
                 </button>
              </div>
            </div>
            <div className="p-4 flex justify-center bg-black/5 relative min-h-75">
              <Image 
                src={imgUrl} 
                alt="원본" 
                width={800} 
                height={600} 
                className="max-h-[65vh] w-auto h-auto object-contain rounded-lg shadow-2xl"
                unoptimized
                priority 
              />
            </div>
          </div>
        </div>
      )}
    </>
  );
}

// --- 메인 패널 ---
export default function Row4ActionPanel({ isDarkMode = true }: Row4Props) {
  // listType에 'public' 추가
  const [listType, setListType] = useState<"pending" | "completed" | "public" | null>(null);
  const [memos, setMemos] = useState<MemoItem[]>([]);
  const [publicMemos, setPublicMemos] = useState<PublicItem[]>([]); // 시민 게시글 상태
  const [loading, setLoading] = useState(false);
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editContent, setEditContent] = useState("");

  // 관리자 메모 페칭 (기존 유지)
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
        const listData = result.dataList[0];
        const fetchedItems = listData.items || [];
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

  // [추가] 시민 게시글 페칭
  const fetchPublicMemos = useCallback(async () => {
    setLoading(true);
    setPublicMemos([]);
    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/public/get?page=0&size=20`, {
        headers: { "Authorization": `Bearer ${localStorage.getItem("accessToken")}` }
      });
      const result: ApiResponse<PublicItem> = await res.json();
      if (result.success && result.dataList) {
        setPublicMemos(result.dataList);
      }
    } catch (e) {
      console.error("Public Fetch Error:", e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (listType === "pending" || listType === "completed") fetchMemos(listType);
    if (listType === "public") fetchPublicMemos();
  }, [listType, fetchMemos, fetchPublicMemos]);

  useEffect(() => {
    const handleRefresh = () => { if (listType === "pending") fetchMemos("pending"); };
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
        headers: isModify ? { "Authorization": `Bearer ${token}` } : { "Content-Type": "application/json", "Authorization": `Bearer ${token}` },
        body
      });
      const result = await res.json();
      if (result.success) {
        setEditingId(null);
        if (listType && listType !== "public") fetchMemos(listType as "pending" | "completed");
      }
    } catch (e) { console.error(e); }
  };

  const theme = {
    modalBg: isDarkMode ? "bg-slate-900 border-white/10" : "bg-white border-slate-200",
    modalHeader: isDarkMode ? "bg-white/2 border-white/5" : "bg-slate-50 border-slate-100",
    listBg: isDarkMode ? "bg-slate-900/50" : "bg-slate-50/50",
    card: isDarkMode ? "bg-white/3 border-white/5 text-slate-200" : "bg-white border-slate-200 text-slate-800 shadow-sm",
    inputText: isDarkMode ? "text-white bg-slate-950" : "text-slate-900 bg-white",
    footerText: isDarkMode ? "text-slate-500 border-white/5" : "text-slate-400 border-slate-100",
  };

  return (
    <div className="grid grid-cols-3 gap-4 h-full"> {/* 3컬럼으로 확장 */}
      {/* 버튼 1: 조치 예정 */}
      <button 
        onClick={() => setListType("pending")} 
        className={`rounded-4xl flex flex-col items-center justify-center gap-3 transition-all group border ${
          isDarkMode ? "bg-blue-600/10 border-blue-500/20 hover:bg-blue-600/20" : "bg-blue-50 border-blue-100 hover:bg-blue-100/50"
        }`}
      >
        <div className={`p-4 rounded-2xl group-hover:scale-110 transition-transform ${isDarkMode ? 'bg-blue-500/20' : 'bg-white shadow-sm'}`}>
          <ListChecks className="w-10 h-10 text-blue-500" />
        </div>
        <span className={`text-sm font-black transition-colors ${isDarkMode ? 'text-blue-100' : 'text-blue-700'}`}>조치 예정 목록</span>
      </button>

      {/* 버튼 2: 조치 완료 */}
      <button 
        onClick={() => setListType("completed")} 
        className={`rounded-4xl flex flex-col items-center justify-center gap-3 transition-all group border ${
          isDarkMode ? "bg-emerald-600/10 border-emerald-500/20 hover:bg-emerald-600/20" : "bg-emerald-50 border-emerald-100 hover:bg-emerald-100/50"
        }`}
      >
        <div className={`p-4 rounded-2xl group-hover:scale-110 transition-transform ${isDarkMode ? 'bg-emerald-500/20' : 'bg-white shadow-sm'}`}>
          <History className="w-10 h-10 text-emerald-500" />
        </div>
        <span className={`text-sm font-black transition-colors ${isDarkMode ? 'text-emerald-100' : 'text-emerald-700'}`}>조치 완료 기록</span>
      </button>

      {/* 버튼 3: 시민 게시판 (새로 추가) */}
      <button 
        onClick={() => setListType("public")} 
        className={`rounded-4xl flex flex-col items-center justify-center gap-3 transition-all group border ${
          isDarkMode ? "bg-purple-600/10 border-purple-500/20 hover:bg-purple-600/20" : "bg-purple-50 border-purple-100 hover:bg-purple-100/50"
        }`}
      >
        <div className={`p-4 rounded-2xl group-hover:scale-110 transition-transform ${isDarkMode ? 'bg-purple-500/20' : 'bg-white shadow-sm'}`}>
          <MessageSquareQuote className="w-10 h-10 text-purple-500" />
        </div>
        <span className={`text-sm font-black transition-colors ${isDarkMode ? 'text-purple-100' : 'text-purple-700'}`}>시민 게시판</span>
      </button>

      {/* 리스트 모달 */}
      {listType && (
        <div className="fixed inset-0 z-100 flex items-center justify-center p-6 bg-black/60 backdrop-blur-sm">
          <div className={`w-full max-w-4xl h-[85vh] flex flex-col shadow-2xl overflow-hidden animate-in slide-in-from-bottom-12 duration-300 rounded-[48px] border ${theme.modalBg}`}>
            {/* 헤더 */}
            <div className={`p-8 border-b flex justify-between items-center ${theme.modalHeader}`}>
              <div className="flex items-center gap-4">
                <div className={`p-3 rounded-2xl ${
                  listType === 'public' ? 'bg-purple-500/20 text-purple-500' :
                  listType === 'pending' ? 'bg-blue-500/20 text-blue-500' : 'bg-emerald-500/20 text-emerald-500'
                }`}>
                  {listType === 'public' ? <MessageSquareQuote className="w-6 h-6"/> : listType === 'pending' ? <ListChecks className="w-6 h-6"/> : <History className="w-6 h-6"/>}
                </div>
                <h2 className={`text-xl font-black ${isDarkMode ? 'text-white' : 'text-slate-900'}`}>
                  {listType === 'public' ? '시민 게시판 (민원 제보)' : listType === 'pending' ? '조치 예정 목록' : '조치 완료 기록'}
                </h2>
              </div>
              <button onClick={() => setListType(null)} className={`p-3 rounded-full transition-colors ${isDarkMode ? 'hover:bg-white/10 text-slate-400' : 'hover:bg-slate-200 text-slate-500'}`}>
                <X className="w-7 h-7"/>
              </button>
            </div>

            {/* 리스트 영역 */}
            <div className={`flex-1 overflow-y-auto p-8 space-y-5 custom-scrollbar ${theme.listBg}`}>
              {loading ? (
                <div className="flex flex-col items-center justify-center h-full gap-4 text-slate-500">
                  <Loader2 className="w-10 h-10 animate-spin text-blue-500" />
                  <p>데이터 로딩 중...</p>
                </div>
              ) : (listType === "public" ? publicMemos : memos).length === 0 ? (
                <div className="flex flex-col items-center justify-center h-full text-slate-400">
                  <p className="text-sm font-medium">기록된 내용이 없습니다.</p>
                </div>
              ) : listType === "public" ? (
                // --- 시민 게시판 렌더링 영역 ---
                publicMemos.map((item) => (
                  <div key={item.no} className={`rounded-4xl p-6 group border transition-all ${theme.card}`}>
                    <div className="flex justify-between items-start gap-8">
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-3">
                          <div className="px-3 py-1 bg-purple-500/10 rounded-full flex items-center gap-2">
                            <MapPin className="w-3.5 h-3.5 text-purple-500" />
                            <span className="text-xs font-bold text-purple-500">{item.pos}</span>
                          </div>
                        </div>
                        <p className={`text-base leading-relaxed whitespace-pre-wrap font-medium ${isDarkMode ? 'text-slate-200' : 'text-slate-700'}`}>{item.content}</p>
                        {item.picture && (
                          <div className="mt-4 rounded-2xl overflow-hidden border border-white/10 w-fit">
                            <Image 
                              src={`data:image/jpeg;base64,${item.picture}`} 
                              alt="시민제보" 
                              width={320} 
                              height={240} 
                              className="object-cover" 
                              unoptimized 
                            />
                          </div>
                        )}
                        <div className={`mt-6 pt-4 border-t text-[11px] font-bold ${theme.footerText}`}>
                          <span>제보 번호: {item.no}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                ))
              ) : (
                // --- 기존 관리자 메모 렌더링 영역 ---
                memos.map((item) => (
                  <div key={item.memoNo} className={`rounded-4xl p-6 group border transition-all ${theme.card}`}>
                    {editingId === item.memoNo ? (
                      <div className="space-y-4">
                        <textarea 
                          className={`w-full border border-blue-500/30 rounded-2xl p-4 text-sm min-h-30 outline-none focus:ring-2 focus:ring-blue-500/20 ${theme.inputText}`} 
                          value={editContent} 
                          onChange={(e) => setEditContent(e.target.value)} 
                        />
                        <div className="flex justify-end gap-3">
                          <button onClick={() => setEditingId(null)} className="px-5 py-2 text-xs text-slate-400">취소</button>
                          <button onClick={() => handleAction('modify', item.memoNo, editContent)} className="px-8 py-2.5 bg-blue-600 text-white rounded-xl text-xs font-black shadow-lg shadow-blue-500/30">저장</button>
                        </div>
                      </div>
                    ) : (
                      <div className="flex justify-between items-start gap-8">
                        <div className="flex-1">
                          <p className={`text-base leading-relaxed whitespace-pre-wrap font-medium ${isDarkMode ? 'text-slate-200' : 'text-slate-700'}`}>{item.content}</p>
                          {item.fileName && <MemoImage memoNo={item.memoNo} fileName={item.fileName} isDarkMode={isDarkMode} />}
                          <div className={`flex items-center gap-6 mt-6 pt-4 border-t text-[11px] font-bold ${theme.footerText}`}>
                            <span className="flex items-center gap-2"><User className="w-3.5 h-3.5 text-blue-500"/> {item.createMember.userName}</span>
                            <span className="flex items-center gap-2"><Clock className="w-3.5 h-3.5 text-slate-400"/> {new Date(item.createTime).toLocaleString()}</span>
                          </div>
                        </div>
                        <div className="flex flex-col gap-2 opacity-0 group-hover:opacity-100 transition-all scale-95 group-hover:scale-100">
                          {listType === "pending" && (
                            <>
                              <button onClick={() => handleAction('disable', item.memoNo)} className="p-3 bg-emerald-500/10 text-emerald-600 hover:bg-emerald-500 hover:text-white rounded-2xl transition-all shadow-sm">
                                <CheckCircle2 className="w-6 h-6" />
                              </button>
                              <button onClick={() => { setEditingId(item.memoNo); setEditContent(item.content); }} className="p-3 bg-blue-500/10 text-blue-600 hover:bg-blue-500 hover:text-white rounded-2xl transition-all shadow-sm">
                                <PencilLine className="w-6 h-6" />
                              </button>
                            </>
                          )}
                          <button onClick={() => handleAction('delete', item.memoNo)} className="p-3 bg-red-500/10 text-red-600 hover:bg-red-500 hover:text-white rounded-2xl transition-all shadow-sm">
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