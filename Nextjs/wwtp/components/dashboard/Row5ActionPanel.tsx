"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import Image from "next/image";
import { Send, AlertCircle, CheckCircle2, Trash2, User, Edit2, X, Save, Paperclip, ImageIcon } from "lucide-react";

interface Memo {
  memoNo: number;
  content: string;
  createTime: string;
  fileName: string | null;
  imageData?: string; 
  createMember: {
    userId: string;
    userName: string;
  };
}

export default function Row5ActionPanel() {
  // --- States ---
  const [memoInput, setMemoInput] = useState("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const [memos, setMemos] = useState<Memo[]>([]);
  const [loading, setLoading] = useState(false);

  // 수정 관련 State
  const [editingMemoNo, setEditingMemoNo] = useState<number | null>(null);
  const [editContent, setEditContent] = useState("");
  const [editFile, setEditFile] = useState<File | null>(null);
  const [editPreviewUrl, setEditPreviewUrl] = useState<string | null>(null);
  const editFileInputRef = useRef<HTMLInputElement>(null);

  // 이미지 확대 모달 State
  const [zoomedImage, setZoomedImage] = useState<string | null>(null);

  // --- Auth & Fetching ---
  const getAuthHeaders = useCallback((isFormData = false): HeadersInit => {
    const token = localStorage.getItem("accessToken");
    const headers: Record<string, string> = {};
    if (!isFormData) headers["Content-Type"] = "application/json";
    if (token) {
      const cleanToken = token.startsWith("Bearer ") ? token.replace("Bearer ", "") : token;
      headers["Authorization"] = `Bearer ${cleanToken.trim()}`;
    }
    return headers;
  }, []);

  const fetchImageData = async (memoNo: number) => {
    try {
      const res = await fetch(`/api/board/memo/image?memo_no=${memoNo}`, {
        headers: getAuthHeaders(),
      });
      const result = await res.json();
      if (result.success && result.dataList) {
        const mimeType = result.dataList[1];
        const base64Data = result.dataList[2];
        return `data:${mimeType};base64,${base64Data}`;
      }
    } catch (err) {
      console.error("이미지 로드 실패:", err);
    }
    return null;
  };

  const fetchMemos = useCallback(async () => {
    try {
      const res = await fetch("/api/board/memo/list?page=0&count=10", {
        headers: getAuthHeaders(),
      });
      const result = await res.json();
      if (result.success && result.dataList?.[0]?.items) {
        const rawMemos = result.dataList[0].items;
        
        const memosWithImages = await Promise.all(rawMemos.map(async (memo: Memo) => {
          if (memo.fileName) {
            const imageData = await fetchImageData(memo.memoNo);
            return { ...memo, imageData };
          }
          return memo;
        }));
        
        setMemos(memosWithImages);
      }
    } catch (err) {
      console.error("메모 로드 실패:", err);
    }
  }, [getAuthHeaders]);

  useEffect(() => {
    fetchMemos();
  }, [fetchMemos]);

  // --- Handlers ---
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>, isEdit: boolean) => {
    const file = e.target.files?.[0] || null;
    if (file) {
      const url = URL.createObjectURL(file);
      if (isEdit) {
        setEditFile(file);
        setEditPreviewUrl(url);
      } else {
        setSelectedFile(file);
        setPreviewUrl(url);
      }
    }
  };

  const handleRegister = async () => {
    if (!memoInput.trim()) return;
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append("content", memoInput);
      if (selectedFile) formData.append("file", selectedFile);

      const res = await fetch("/api/board/memo/create", {
        method: "PUT",
        headers: getAuthHeaders(true),
        body: formData,
      });
      
      const result = await res.json();
      if (result.success) {
        setMemoInput("");
        setSelectedFile(null);
        setPreviewUrl(null);
        if (fileInputRef.current) fileInputRef.current.value = "";
        fetchMemos();
      } else {
        alert(result.errorMsg);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleUpdate = async (memoNo: number) => {
    if (!editContent.trim()) return;
    try {
      const formData = new FormData();
      formData.append("memoNo", memoNo.toString());
      formData.append("content", editContent);
      if (editFile) formData.append("file", editFile);

      const res = await fetch("/api/board/memo/modify", {
        method: "POST",
        headers: getAuthHeaders(true),
        body: formData,
      });
      
      const result = await res.json();
      if (result.success) {
        setEditingMemoNo(null);
        setEditFile(null);
        setEditPreviewUrl(null);
        fetchMemos();
      }
    } catch (err) {
      console.error("수정 실패:", err);
    }
  };

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

  const closeZoom = () => setZoomedImage(null);

  return (
    <div className="bg-[#1a202c] p-6 rounded-4xl border border-white/5 h-full flex flex-col shadow-2xl relative">
      {/* 상단 헤더 */}
      <div className="shrink-0 mb-4">
        <h3 className="text-[16px] font-bold text-blue-400 flex items-center gap-2 mb-4">
          <AlertCircle className="w-5 h-5 text-blue-500" /> 운영 권고 & 조치
        </h3>
        <div className="bg-blue-500/10 px-4 py-3 rounded-2xl border border-blue-500/20 text-sm text-blue-100 flex items-center gap-3">
          <span className="text-lg">💡</span> 송풍량 +10% 권장 (질산화 저하 위험)
        </div>
      </div>

      {/* 등록 입력부 */}
      <div className="shrink-0 space-y-3 mb-6">
        <div className="flex gap-2">
          <input 
            type="text"
            value={memoInput}
            onChange={(e) => setMemoInput(e.target.value)}
            placeholder="조치 사항 입력..."
            onKeyDown={(e) => e.key === 'Enter' && handleRegister()}
            className="flex-1 bg-slate-800/50 border border-white/10 rounded-2xl px-5 py-3 text-sm text-slate-200 focus:border-blue-500 outline-none transition-all placeholder:text-slate-600"
          />
          <input type="file" ref={fileInputRef} onChange={(e) => handleFileChange(e, false)} className="hidden" accept="image/*" />
          <button 
            onClick={() => fileInputRef.current?.click()}
            className={`p-3 rounded-2xl border border-white/10 transition-colors ${selectedFile ? 'text-blue-400 bg-blue-500/20 border-blue-500/50' : 'text-slate-400 hover:bg-white/5'}`}
          >
            <Paperclip className="w-5 h-5" />
          </button>
          <button onClick={handleRegister} disabled={loading} className="bg-blue-600 hover:bg-blue-500 px-6 rounded-2xl font-bold text-sm text-white flex items-center gap-2 shadow-lg disabled:opacity-50 transition-all">
            <Send className="w-4 h-4" /> 등록
          </button>
        </div>

        {previewUrl && (
          <div className="flex flex-col gap-1.5">
            <div className="relative w-20 h-20 rounded-2xl overflow-hidden border border-blue-500/50 group shadow-lg">
              <Image src={previewUrl} alt="preview" fill className="object-cover" unoptimized />
              <button 
                onClick={() => { setSelectedFile(null); setPreviewUrl(null); if(fileInputRef.current) fileInputRef.current.value=""; }}
                className="absolute inset-0 bg-black/40 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
              >
                <X className="w-5 h-5 text-white" />
              </button>
            </div>
            <p className="text-[11px] text-slate-400 w-20 truncate px-1" title={selectedFile?.name}>
              {selectedFile?.name}
            </p>
          </div>
        )}
      </div>

      <div className="border-t border-white/5 mb-4 shrink-0"></div>

      {/* 메모 리스트 */}
      <div className="flex-1 min-h-0 overflow-y-auto pr-2 space-y-3 custom-scrollbar">
        {memos.length === 0 ? (
          <div className="h-full flex items-center justify-center text-slate-600 text-sm italic">등록된 조치 이력이 없습니다.</div>
        ) : (
          memos.map((memo) => (
            <div key={memo.memoNo} className="bg-slate-800/30 p-4 rounded-2xl border border-white/5 hover:border-white/10 transition-all group">
              {editingMemoNo === memo.memoNo ? (
                /* 수정 모드 */
                <div className="space-y-3">
                  <textarea 
                    value={editContent}
                    onChange={(e) => setEditContent(e.target.value)}
                    className="w-full bg-slate-900 border border-blue-500/50 rounded-xl p-3 text-sm text-white outline-none"
                    rows={2}
                  />
                  <div className="flex items-center gap-4">
                    <input type="file" ref={editFileInputRef} onChange={(e) => handleFileChange(e, true)} className="hidden" accept="image/*" />
                    <button onClick={() => editFileInputRef.current?.click()} className="flex items-center gap-2 text-xs text-slate-400 bg-white/5 px-3 py-2 rounded-lg hover:bg-white/10">
                      <ImageIcon className="w-4 h-4" /> {editFile ? "변경됨" : "사진 수정"}
                    </button>
                    {editPreviewUrl && (
                      <div className="flex flex-col gap-1">
                        <div className="relative w-10 h-10">
                          <Image src={editPreviewUrl} alt="edit preview" fill className="rounded-lg object-cover border border-white/10" unoptimized />
                        </div>
                        <p className="text-[10px] text-slate-500 w-16 truncate text-center">
                          {editFile ? editFile.name : memo.fileName}
                        </p>
                      </div>
                    )}
                    <div className="flex-1" />
                    <button onClick={() => handleUpdate(memo.memoNo)} className="p-2 rounded-xl bg-blue-600 text-white hover:bg-blue-500 transition-colors"><Save className="w-4 h-4" /></button>
                    <button onClick={() => { setEditingMemoNo(null); setEditPreviewUrl(null); }} className="p-2 rounded-xl bg-slate-700 text-white hover:bg-slate-600 transition-colors"><X className="w-4 h-4" /></button>
                  </div>
                </div>
              ) : (
                /* 일반 모드 */
                <>
                  <div className="flex gap-4">
                    <div className="flex-1">
                      <p className="text-sm text-slate-200 leading-relaxed break-all">{memo.content}</p>
                    </div>
                    {memo.imageData && (
                      <div className="shrink-0 flex flex-col items-center gap-1.5">
                        <div className="relative w-16 h-16 shadow-inner">
                          <Image 
                            src={memo.imageData} 
                            alt="memo attach" 
                            fill
                            className="rounded-2xl object-cover border border-white/10 hover:scale-105 transition-transform cursor-pointer"
                            onClick={() => setZoomedImage(memo.imageData || null)}
                            unoptimized
                          />
                        </div>
                        <span className="text-[10px] text-slate-500 w-16 truncate text-center font-medium" title={memo.fileName || ""}>
                          {memo.fileName}
                        </span>
                      </div>
                    )}
                  </div>

                  <div className="flex justify-between items-end mt-4">
                    <div className="flex flex-col gap-1">
                      <span className="text-xs text-slate-500 flex items-center gap-1.5"><User className="w-3 h-3" /> {memo.createMember?.userName}</span>
                      <span className="text-[11px] text-slate-600 font-mono tracking-tighter">
                        {new Date(memo.createTime).toLocaleString('ko-KR', {
                          year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit'
                        })}
                      </span>
                    </div>
                    <div className="flex gap-1.5 opacity-0 group-hover:opacity-100 transition-opacity">
                      <button 
                        onClick={() => { 
                          setEditingMemoNo(memo.memoNo); 
                          setEditContent(memo.content);
                          setEditPreviewUrl(memo.imageData || null);
                        }} 
                        className="p-2 rounded-xl bg-white/5 hover:bg-white/10 text-slate-400"
                      ><Edit2 className="w-4 h-4" /></button>
                      <button onClick={() => handleComplete(memo.memoNo)} className="p-2 rounded-xl bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-500"><CheckCircle2 className="w-4 h-4" /></button>
                      <button onClick={() => handleDelete(memo.memoNo)} className="p-2 rounded-xl bg-red-500/10 hover:bg-red-500/20 text-red-400"><Trash2 className="w-4 h-4" /></button>
                    </div>
                  </div>
                </>
              )}
            </div>
          ))
        )}
      </div>

      {/* --- 이미지 확대 모달 영역 --- */}
      {zoomedImage && (
        <div 
          className="fixed inset-0 z-100 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4 animate-in fade-in duration-200"
          onClick={closeZoom}
        >
          <div 
            className="relative max-w-5xl max-h-[85vh] w-full md:w-1/2 overflow-hidden rounded-3xl border border-white/10 shadow-2xl bg-slate-900 flex items-center justify-center"
            onClick={(e) => e.stopPropagation()}
          >
            <button 
              onClick={closeZoom}
              className="absolute top-4 right-4 z-10 p-2 bg-black/50 hover:bg-black/80 rounded-full text-white transition-colors"
            >
              <X className="w-6 h-6" />
            </button>
            
            <div className="relative w-full h-[80vh] p-2">
               <Image 
                src={zoomedImage} 
                alt="Enlarged view" 
                fill
                className="object-contain rounded-xl"
                unoptimized
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}