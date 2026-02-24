"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import useSWR from "swr";

// 분리한 컴포넌트들
import Row1Status from "@/components/dashboard/Row1Status";      
import Row2Charts from "@/components/dashboard/Row2Charts";      
import Row3RiskDetail from "@/components/dashboard/Row3RiskDetail"; 
import Row4Alerts from "@/components/dashboard/Row4Alerts";      
import Row5ActionPanel from "@/components/dashboard/Row5ActionPanel"; 

import EditProfileModal from "../options/EditProfileModal";

// SWR 페처 함수
const fetcher = async (url: string) => {
  const token = typeof window !== "undefined" ? localStorage.getItem("accessToken") : null;
  const response = await fetch(url, {
    method: "GET",
    headers: { 
      "Content-Type": "application/json", 
      ...(token ? { "Authorization": `Bearer ${token}` } : {}) 
    },
  });
  if (response.status === 401) throw new Error("인증 세션이 만료되었습니다.");
  if (!response.ok) throw new Error("API 연결 실패");
  return response.json();
};

interface HealthResponse {
  success: boolean;
  message?: string;
  checkTime?: string;
}

export default function DashboardPage() {
  const router = useRouter();
  const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL;

  const [isAuthChecked, setIsAuthChecked] = useState(false);
  const [userData, setUserData] = useState({
    userNo: 0,
    userId: "",
    userName: "",
    userRole: "",
    userEmail: ""
  });
  const [isProfileOpen, setIsProfileOpen] = useState(false);
  const [isEditModalOpen, setIsEditModalOpen] = useState(false);
  const [errorStartTime, setErrorStartTime] = useState<string | null>(null);

  // --- 시스템 헬스 체크 데이터 페칭 (30초 간격) ---
  const { data: healthRaw, error: healthError } = useSWR<HealthResponse>(
    `${API_BASE_URL}/api/board/health`, 
    fetcher, 
    { refreshInterval: 30000 }
  );

  const isSystemOk = !!healthRaw && healthRaw.success === true && !healthError;

  // 에러 발생 시각 관리 로직
  useEffect(() => {
    const isCurrentlyDown = !!healthError || (healthRaw && !healthRaw.success);
    if (isCurrentlyDown) {
      if (!errorStartTime) {
        const now = new Date();
        const formatted = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')} ${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}`;
        setErrorStartTime(healthRaw?.checkTime || formatted);
      }
    } else if (healthRaw?.success) {
      setErrorStartTime(null);
    }
  }, [healthRaw, healthError, errorStartTime]);

  // 인증 및 로컬스토리지 데이터 로드
  useEffect(() => {
    const savedRole = localStorage.getItem('userRole');
    if (!savedRole) {
      router.replace("/");
      return;
    }
    
    setUserData({
      userNo: Number(localStorage.getItem('userNo')),
      userId: localStorage.getItem('userId') || "",
      userName: localStorage.getItem('userName') || "사용자",
      userRole: savedRole,
      userEmail: localStorage.getItem('userEmail') || "" 
    });

    setIsAuthChecked(true);
  }, [router]);

  const handleLogout = () => {
    if (confirm("로그아웃 하시겠습니까?")) {
      localStorage.clear();
      router.push("/");
    }
  };

  if (!isAuthChecked) return <div className="min-h-screen bg-slate-950" />;

  return (
    <div className="min-h-screen lg:h-screen bg-[#0a0f1d] text-slate-200 p-5 font-sans flex flex-col lg:overflow-hidden selection:bg-blue-500/30">
      
      {/* --- 상단 헤더 --- */}
      <header className="flex justify-between items-center mb-6 border-b border-white/5 pb-5 h-14 flex-none relative">
        <div className="z-10">
          <h1 className="text-xl font-black text-white flex items-center gap-2.5 tracking-tight">
            <span className={`w-3 h-3 rounded-full transition-all duration-500 shadow-lg ${isSystemOk ? "bg-blue-500 shadow-blue-500/80" : "bg-red-500 shadow-red-500/80"}`}></span>
            Smart WWTP <span className="text-slate-400 font-light">Monitoring</span>
          </h1>
          <p className="text-slate-500 text-[9px] uppercase tracking-[0.2em] font-bold mt-1">Integrated Operation Center</p>
        </div>

        {/* --- 중앙 시스템 상태 표시부 (날짜 제거 및 문구 수정) --- */}
        <div className="hidden md:block absolute left-1/2 transform -translate-x-1/2">
          <div className={`px-6 py-2 rounded-2xl backdrop-blur-md border transition-all duration-500 flex items-center gap-4 shadow-2xl ${
            isSystemOk 
              ? "bg-emerald-500/5 border-emerald-500/20 shadow-emerald-500/5" 
              : "bg-red-500/10 border-red-500/40 shadow-red-500/10"
          }`}>
            <div className="flex items-center gap-3">
              <span className="text-slate-400 text-xs font-bold tracking-tight">시스템 상태 :</span>
              <div className="flex items-center gap-2">
                <span className={`text-sm font-black tracking-tight ${isSystemOk ? "text-emerald-400" : "text-red-400"}`}>
                  {isSystemOk ? "정상" : "비정상"}
                </span>
                <span className="relative flex h-2 w-2">
                  <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${isSystemOk ? "bg-emerald-400" : "bg-red-500"}`}></span>
                  <span className={`relative inline-flex rounded-full h-2 w-2 ${isSystemOk ? "bg-emerald-500" : "bg-red-600"}`}></span>
                </span>
              </div>
            </div>

            {/* 비정상 시 감지 시간 표시 (세로 구분선 포함) */}
            {!isSystemOk && errorStartTime && (
              <>
                <div className="w-px h-3 bg-white/10" />
                <div className="flex items-center gap-2 px-3 py-1 rounded-lg bg-red-500/10 border border-red-500/20">
                  <span className="text-[10px] font-bold text-red-300 uppercase tracking-tighter">⚠️ 감지시간 :</span>
                  <span className="text-xs font-mono font-bold text-red-200">{errorStartTime}</span>
                </div>
              </>
            )}
          </div>
        </div>

        <div className="relative flex items-center gap-5 z-10">
          <div 
            onClick={() => setIsProfileOpen(!isProfileOpen)} 
            className="flex items-center gap-3 hover:bg-white/5 px-3 py-1.5 rounded-2xl transition-all cursor-pointer border border-transparent hover:border-white/10 group"
          >
            <div className="text-right hidden sm:block"> 
              <p className="text-sm font-bold text-slate-100 group-hover:text-white transition-colors">{userData.userName}님</p>
              <p className="text-[10px] text-blue-500 font-bold uppercase tracking-widest">
                {userData.userRole.replace("ROLE_", "")}
              </p>
            </div>
            <div className="w-9 h-9 bg-linear-to-tr from-blue-600 to-violet-600 rounded-xl flex items-center justify-center text-sm font-black shadow-lg shadow-blue-500/20 transform group-hover:rotate-3 transition-transform">
              {userData.userName.substring(0, 1)}
            </div>
          </div>

          <button onClick={handleLogout} className="px-4 py-1.5 rounded-xl bg-red-500/5 hover:bg-red-500/20 hover:text-red-400 border border-red-500/10 transition-all text-xs font-bold text-slate-400">
            LOGOUT
          </button>
          
          <AnimatePresence>
            {isProfileOpen && (
              <motion.div 
                initial={{ opacity: 0, y: 10, scale: 0.95 }} 
                animate={{ opacity: 1, y: 0, scale: 1 }} 
                exit={{ opacity: 0, y: 10, scale: 0.95 }} 
                className="absolute right-0 top-full mt-4 w-72 bg-slate-900/95 backdrop-blur-xl rounded-2xl z-50 overflow-hidden border border-white/10 shadow-[0_20px_50px_rgba(0,0,0,0.5)]"
              >
                <div className="p-8 flex flex-col items-center border-b border-white/5 bg-linear-to-b from-white/5 to-transparent">
                  <div className="w-20 h-20 bg-linear-to-tr from-blue-600 to-indigo-600 rounded-2xl flex items-center justify-center text-white text-2xl font-black shadow-2xl mb-4">
                    {userData.userName.substring(0, 1)}
                  </div>
                  <p className="font-bold text-xl text-white leading-tight">{userData.userName}</p>
                  <p className="text-sm text-slate-500 mt-1">{userData.userId}</p>
                  <p className="text-[11px] text-slate-400 mt-1 break-all px-4 text-center">{userData.userEmail}</p>
                  <span className="mt-3 px-3 py-0.5 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-400 text-[10px] font-black uppercase tracking-widest">
                    {userData.userRole}
                  </span>
                </div>
                <div className="flex flex-col text-sm p-3 gap-1">
                  <button onClick={() => { setIsEditModalOpen(true); setIsProfileOpen(false); }} className="flex items-center gap-3 px-4 py-3 rounded-xl hover:bg-white/5 transition-all text-slate-300 font-medium">
                    <span className="text-lg text-slate-500">👤</span> 개인정보 수정
                  </button>
                  {userData.userRole === "ROLE_ADMIN" && (
                    <>
                      <button onClick={() => { router.push("/admin/member"); setIsProfileOpen(false); }} className="flex items-center gap-3 px-4 py-3 rounded-xl hover:bg-white/5 transition-all text-slate-300 font-medium">
                        <span className="text-lg text-slate-500">⚙️</span> 사원 관리
                      </button>
                      <button onClick={() => { router.push("/admin/memo-history"); setIsProfileOpen(false); }} className="flex items-center gap-3 px-4 py-3 rounded-xl hover:bg-white/5 transition-all text-slate-300 font-medium">
                        <span className="text-lg text-slate-500">📜</span> 조치 이력 기록
                      </button>
                    </>
                  )}
                  <div className="my-1 border-t border-white/5" />
                  <button onClick={handleLogout} className="flex items-center gap-3 px-4 py-3 rounded-xl hover:bg-red-500/10 text-red-400 transition-all font-bold">
                    <span className="text-lg">🚪</span> LOGOUT
                  </button>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </header>

      {/* --- 대시보드 메인 콘텐츠 --- */}
      <main className="flex-1 grid grid-cols-10 gap-6 lg:overflow-hidden min-h-0">
        <section className="col-span-12 lg:col-span-6 flex flex-col gap-6 min-h-0">
          <div id="row1-container" className="flex-none drop-shadow-sm">
            <Row1Status />
          </div>
          <div className="flex-1 lg:min-h-0 bg-white/2 border border-white/5 rounded-4xl overflow-hidden shadow-inner">
            <Row2Charts />
          </div>
        </section>

        <section className="col-span-12 lg:col-span-4 flex flex-col gap-6 min-h-0">
          <div className="flex-none bg-white/2 border border-white/5 rounded-4xl overflow-hidden">
            <Row3RiskDetail />
          </div>
          <div className="flex-none bg-white/2 border border-white/5 rounded-4xl overflow-hidden">
            <Row4Alerts />
          </div>
          <div className="flex-1 lg:min-h-0 bg-white/2 border border-white/5 rounded-4xl overflow-hidden">
            <Row5ActionPanel />
          </div>
        </section>
      </main>

      {/* 개인정보 수정 모달 */}
      <EditProfileModal 
        isOpen={isEditModalOpen} 
        onClose={() => setIsEditModalOpen(false)} 
        currentUser={{ 
          userNo: userData.userNo, 
          id: userData.userId, 
          name: userData.userName, 
          role: userData.userRole,
          email: userData.userEmail 
        }} 
        onUpdateSuccess={(newId, newName, newEmail) => { 
          setUserData(prev => ({ 
            ...prev, 
            userId: newId, 
            userName: newName,
            userEmail: newEmail || prev.userEmail 
          })); 
          localStorage.setItem('userId', newId);
          localStorage.setItem('userName', newName);
          if (newEmail) localStorage.setItem('userEmail', newEmail);
        }} 
      />
    </div>
  );
}