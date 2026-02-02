"use client";

import { useState, useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import EditProfileModal from "../components/EditProfileModal";
import AddMemberModal from "../components/AddMemberModal";

interface ProcessDataItem {
  time: string;
  toc: number;
  ph: number;
  ss: number;
  flux: number;
  tn: number;
  tp: number;
}

interface Memo {
  memoNo: number;
  content: string;
  createTime: string; // 백엔드 엔티티와 일치
  createMember: {
    userId: string;
    userName: string;
  };
}

const MetricCard = ({ title, value, unit, color, onClick }: any) => (
  <motion.div
    whileHover={{ scale: 1.02, translateY: -5 }}
    whileTap={{ scale: 0.98 }}
    onClick={onClick}
    className="bg-slate-800/40 p-8 rounded-3xl border border-white/5 backdrop-blur-md cursor-pointer hover:border-blue-500/50 transition-all shadow-2xl flex flex-col justify-between min-h-45"
  >
    <div>
      <div className="flex items-center gap-2 mb-4">
        <div className={`w-2 h-2 rounded-full animate-pulse`} style={{ backgroundColor: color }}></div>
        <h3 className="text-slate-400 font-medium tracking-wider">{title}</h3>
      </div>
      <div className="flex items-baseline gap-2">
        <span className="text-4xl font-bold tracking-tight text-white">{value}</span>
        <span className="text-slate-500 font-medium">{unit}</span>
      </div>
    </div>
    <div className="mt-6 flex justify-between items-center text-xs text-slate-500 border-t border-white/5 pt-4">
      <span>실시간 데이터</span>
      <span className="text-blue-400">상세보기 →</span>
    </div>
  </motion.div>
);

export default function DashboardPage() {
  const router = useRouter();
  const [isAuthChecked, setIsAuthChecked] = useState(false);
  const [userNo, setUserNo] = useState<number>(0);
  const [userRole, setUserRole] = useState("");
  const [userId, setUserId] = useState("");
  const [userName, setUserName] = useState("");
  const [isProfileOpen, setIsProfileOpen] = useState(false);
  const [isEditModalOpen, setIsEditModalOpen] = useState(false);
  const [isAddModalOpen, setIsAddModalOpen] = useState(false);
  const [processData, setProcessData] = useState<ProcessDataItem[]>([]);

  const [memos, setMemos] = useState<Memo[]>([]);
  const [oldMemos, setOldMemos] = useState<Memo[]>([]);
  const [newMemoContent, setNewMemoContent] = useState("");
  const [isMemoLoading, setIsMemoLoading] = useState(false);

  // --- 중요: 인증 헤더 생성 및 디버깅 ---
  const getAuthHeaders = useCallback(() => {
    const token = localStorage.getItem("accessToken");
    
    // 콘솔에서 토큰 존재 여부 확인용
    if (!token) {
      console.error("❌ 로컬 스토리지에 accessToken이 없습니다!");
      return { "Content-Type": "application/json"}
    }

    const cleanToken = token.startsWith("Bearer ") ? token.replace("Bearer ", "") : token;

    return {
      "Content-Type": "application/json",
      // Bearer 뒤에 반드시 공백 한 칸이 있어야 합니다.
      "Authorization": token ? `Bearer ${cleanToken}` : "",
    };
  }, []);

  const fetchMemos = useCallback(async () => {
    if (localStorage.getItem("userRole") === "ROLE_VIEWER") return;

    try {
      const headers = getAuthHeaders();
      console.log("요청 헤더 확인:", headers); // 👈 여기서 Bearer 토큰이 잘 찍히는지 확인

      const [resActive, resOld] = await Promise.all([
        fetch("/api/memo/list?page=0&count=10", { headers }),
        fetch("/api/memo/oldList?page=0&count=10", { headers })
      ]);

      // 응답이 JSON인지 HTML인지 먼저 체크
      const ct = resActive.headers.get("content-type");
      if (ct && !ct.includes("application/json")) {
        const text = await resActive.text();
        console.error("⚠️ 서버가 JSON이 아닌 HTML을 보냈습니다. (인증/권한 에러 가능성)");
        console.error("서버 응답 내용 일부:", text.substring(0, 200));
        return;
      }

      const activeResult = await resActive.json();
      const oldResult = await resOld.json();

      if (activeResult.success && activeResult.dataList?.[0]?.items) {
        setMemos(activeResult.dataList[0].items);
      }
      if (oldResult.success && oldResult.dataList?.[0]?.items) {
        setOldMemos(oldResult.dataList[0].items);
      }
    } catch (err) {
      console.error("메모 페칭 중 예외 발생:", err);
    }
  }, [getAuthHeaders]);

  useEffect(() => {
    const savedRole = localStorage.getItem('userRole');
    const savedId = localStorage.getItem('userId');
    const savedName = localStorage.getItem('userName');
    const savedNo = localStorage.getItem('userNo');
    
    if (!savedRole) {
      router.replace("/");
      return;
    }

    setUserNo(Number(savedNo));
    setUserId(savedId || "");
    setUserRole(savedRole);
    setUserName(savedName || "username");
    setIsAuthChecked(true);

    fetch("/data/process_data.json")
      .then(res => res.json())
      .then(json => setProcessData(json))
      .catch(err => console.error("데이터 로딩 실패:", err));

    fetchMemos();
  }, [router, fetchMemos]);

  const handleCreateMemo = async () => {
    if (!newMemoContent.trim()) return;
    setIsMemoLoading(true);
    try {
      const res = await fetch("/api/memo/create", {
        method: "PUT",
        headers: getAuthHeaders(),
        body: JSON.stringify({ content: newMemoContent }),
      });
      const result = await res.json();
      if (result.success) {
        setNewMemoContent("");
        fetchMemos();
      }
    } finally {
      setIsMemoLoading(false);
    }
  };

  const handleDeleteMemo = async (memoNo: number) => {
    if (!confirm("삭제하시겠습니까?")) return;
    const res = await fetch("/api/memo/disable", {
      method: "POST",
      headers: getAuthHeaders(),
      body: JSON.stringify({ memoNo }),
    });
    const result = await res.json();
    if (result.success) fetchMemos();
  };

  const handleLogout = () => {
    if (confirm("로그아웃 하시겠습니까?")) {
      localStorage.clear();
      router.push("/"); 
    }
  };

  if (!isAuthChecked) return <div className="min-h-screen bg-slate-900" />;

  const latest = processData.length > 0 ? processData[processData.length - 1] : null;
  const metrics = [
    { title: "TOC (총유기탄소)", key: "toc", color: "#60a5fa", unit: "mg/L" },
    { title: "pH (수소이온농도)", key: "ph", color: "#34d399", unit: "pH" },
    { title: "SS (부유물질)", key: "ss", color: "#fbbf24", unit: "mg/L" },
    { title: "FLUX (유량)", key: "flux", color: "#a78bfa", unit: "m³/h" },
    { title: "TN (총질소)", key: "tn", color: "#f87171", unit: "mg/L" },
    { title: "TP (총인)", key: "tp", color: "#22d3ee", unit: "mg/L" },
  ];

  return (
    <div className="min-h-screen bg-slate-900 text-white p-8 font-sans">
      <header className="flex justify-between items-center mb-12 border-b border-white/10 pb-8">
        <div>
          <h1 className="text-3xl font-bold text-white flex items-center gap-3">
            <span className="text-blue-500">●</span> Smart WWTP Monitoring
          </h1>
          <p className="text-slate-400 mt-2 font-light">공정별 실시간 상태 요약</p>
        </div>
        <div className="relative flex items-center gap-4">
          <div onClick={() => setIsProfileOpen(!isProfileOpen)} className="flex items-center gap-3 hover:bg-white/5 p-2 rounded-xl transition-all cursor-pointer border border-transparent hover:border-white/10">
            <div className="text-right">
              <p className="text-sm font-bold text-white">{userName}님</p>
              <p className="text-[10px] text-slate-500 uppercase">{userRole.replace("ROLE_", "")}</p>
            </div>
            <div className="w-10 h-10 bg-linear-to-br from-blue-500 to-indigo-600 rounded-full flex items-center justify-center text-sm font-bold shadow-lg">
              {userName.substring(0, 1)}
            </div>
          </div>
          <button onClick={handleLogout} className="px-4 py-2 rounded-lg bg-white/5 hover:bg-red-500/10 hover:text-red-400 border border-white/10 transition-all text-xs font-medium">
            로그아웃
          </button>
          
          <AnimatePresence>
            {isProfileOpen && (
              <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 10 }} className="absolute right-0 top-full mt-4 w-64 bg-slate-700 rounded-2xl z-50 overflow-hidden border border-slate-500/50 shadow-2xl">
                <div className="p-6 flex flex-col items-center border-b border-slate-600 bg-slate-600/50">
                  <div className="w-16 h-16 bg-blue-500 rounded-full flex items-center justify-center text-white text-xl font-bold shadow-lg">{userName.substring(0, 1)}</div>
                  <p className="font-bold text-lg text-white mt-3">{userName}</p>
                  <p className="text-xs text-blue-300/80">{userId}</p>
                </div>
                <div className="flex flex-col text-sm p-2">
                  <button onClick={() => { setIsEditModalOpen(true); setIsProfileOpen(false); }} className="flex items-center gap-3 px-4 py-3 rounded-xl hover:bg-slate-600 transition-all">👤 개인정보 수정</button>
                  {userRole === "ROLE_ADMIN" && <button onClick={() => { router.push("/admin/member"); setIsProfileOpen(false); }} className="flex items-center gap-3 px-4 py-3 rounded-xl hover:bg-slate-600 transition-all">⚙️ 사원 관리</button>}
                  <button onClick={handleLogout} className="flex items-center gap-3 px-4 py-3 rounded-xl hover:bg-red-500/20 text-red-300 transition-all">🚪 <b>로그아웃</b></button>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 mb-16">
        {metrics.map((metric) => (
          <MetricCard key={metric.key} title={metric.title} value={latest ? (latest[metric.key as keyof ProcessDataItem]) : "..."} unit={metric.unit} color={metric.color} onClick={() => router.push(`/dashboard/${metric.key}`)} />
        ))}
      </div>

      {userRole !== "ROLE_VIEWER" && (
        <div className="space-y-12">
          <section className="bg-slate-800/30 p-8 rounded-3xl border border-white/5 backdrop-blur-md">
            <h3 className="text-xl font-bold mb-8 flex items-center gap-3 text-white">
              <span className="text-blue-500">📝</span> 실시간 업무 공유
            </h3>
            <div className="flex gap-4 mb-10">
              <input type="text" value={newMemoContent} onChange={(e) => setNewMemoContent(e.target.value)} placeholder="업무 내용을 입력 후 Enter..." className="flex-1 bg-white/5 border border-white/10 rounded-2xl px-6 py-4 focus:border-blue-500 outline-none transition-all" onKeyDown={(e) => e.key === "Enter" && handleCreateMemo()} />
              <button onClick={handleCreateMemo} disabled={isMemoLoading} className="px-8 py-4 bg-blue-600 hover:bg-blue-500 rounded-2xl font-bold transition-all">{isMemoLoading ? "등록 중..." : "등록"}</button>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {memos.map((memo) => (
                <div key={memo.memoNo} className="relative bg-white/5 p-6 rounded-2xl border border-white/5 hover:border-blue-500/30 transition-all group">
                  <div className="absolute top-4 right-4 flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                    {(userRole === "ROLE_ADMIN" || memo.createMember?.userId === userId) && (
                      <button onClick={() => handleDeleteMemo(memo.memoNo)} className="p-1.5 hover:bg-red-500/20 rounded text-red-400 text-xs">삭제</button>
                    )}
                  </div>
                  <p className="text-slate-200 text-sm mb-6 leading-relaxed pr-8">{memo.content}</p>
                  <div className="flex justify-between items-center pt-4 border-t border-white/5 text-[11px] text-slate-500">
                    <span className="font-medium text-blue-400">{memo.createMember?.userName || "작성자"}</span>
                    <span>{memo.createTime ? new Date(memo.createTime).toLocaleDateString() : "-"}</span>
                  </div>
                </div>
              ))}
            </div>
          </section>

          <section className="bg-slate-900/40 p-8 rounded-3xl border border-white/5 opacity-50">
            <h3 className="text-lg font-bold mb-6 text-slate-400">완료된 업무 이력</h3>
            <div className="grid grid-cols-1 md:grid-cols-4 lg:grid-cols-5 gap-4">
              {oldMemos.map((memo) => (
                <div key={memo.memoNo} className="p-4 bg-white/5 rounded-xl border border-white/5 text-[11px]">
                  <p className="text-slate-400 truncate">{memo.content}</p>
                  <p className="text-slate-600 mt-2">{memo.createMember?.userName || "작성자"}</p>
                </div>
              ))}
            </div>
          </section>
        </div>
      )}

      <EditProfileModal isOpen={isEditModalOpen} onClose={() => setIsEditModalOpen(false)} currentUser={{ userNo, id: userId, name: userName, role: userRole }} onUpdateSuccess={(newId, newName) => { setUserId(newId); setUserName(newName); }} />
      <AddMemberModal isOpen={isAddModalOpen} onClose={() => setIsAddModalOpen(false)} onSuccess={fetchMemos} />
    </div>
  );
}