"use client";

import { useState, useEffect, useMemo } from "react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import useSWR from "swr";

// 컴포넌트 임포트
import Row1Charts from "@/components/dashboard/Row1Charts";      
import Row2RiskDetail from "@/components/dashboard/Row2RiskDetail"; 
import Row3Alerts from "@/components/dashboard/Row3Alerts";         
import Row4ActionPanel from "@/components/dashboard/Row4ActionPanel"; 
import EditProfileModal from "../options/EditProfileModal";

// --- 인터페이스 정의 ---
interface WeatherData { SYS_TIME: string; TA: number; RN_15m: number; HM: number; }
interface TmsRecord { SYS_TIME: string; TOC_VU: number; PH_VU: number; SS_VU: number; FLUX_VU: number; TN_VU: number; TP_VU: number; }
interface FlowRecord { SYS_TIME: string; Q_in: number; }
interface BoardViewResponse { success: boolean; dataList: [TmsRecord[], TmsRecord[], FlowRecord[], FlowRecord[], WeatherData[]]; }
interface HealthResponse { success: boolean; message?: string; checkTime?: string; }

const fetcher = async (url: string) => {
  const token = typeof window !== "undefined" ? localStorage.getItem("accessToken") : null;
  const response = await fetch(url, {
    method: "GET",
    headers: { "Content-Type": "application/json", ...(token ? { "Authorization": `Bearer ${token}` } : {}) },
  });
  if (response.status === 401) throw new Error("인증 세션 만료");
  if (!response.ok) throw new Error("API 연결 실패");
  return response.json();
};

export default function DashboardPage() {
  const router = useRouter();
  const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL;

  const [isAuthChecked, setIsAuthChecked] = useState(false);
  const [userData, setUserData] = useState({
    userNo: 0, userId: "", userName: "", userRole: "", userEmail: ""
  });
  const [isProfileOpen, setIsProfileOpen] = useState(false);
  const [isEditModalOpen, setIsEditModalOpen] = useState(false);

  // --- 테마 상태 관리 (다크 모드 기본) ---
  const [isDarkMode, setIsDarkMode] = useState(true);

  useEffect(() => {
    const savedTheme = localStorage.getItem("theme");
    if (savedTheme === "light") setIsDarkMode(false);
  }, []);

  const toggleTheme = () => {
    const newTheme = !isDarkMode;
    setIsDarkMode(newTheme);
    localStorage.setItem("theme", newTheme ? "dark" : "light");
  };

  // 데이터 페칭 (30초 간격 갱신)
  const { data: healthRaw, error: healthError } = useSWR<HealthResponse>(`${API_BASE_URL}/api/board/health`, fetcher, { refreshInterval: 30000 });
  const { data: boardData } = useSWR<BoardViewResponse>(`${API_BASE_URL}/api/board/boardView`, fetcher, { refreshInterval: 30000 });

  const isSystemOk = !!healthRaw && healthRaw.success === true && !healthError;

  const latestWeather = useMemo(() => {
    if (!boardData?.success || !boardData.dataList[4]) return null;
    const weatherList = boardData.dataList[4];
    return weatherList.length > 0 ? weatherList[weatherList.length - 1] : null;
  }, [boardData]);

  const latestTms = useMemo(() => {
    if (!boardData?.success || !boardData.dataList[0]) return null;
    const tmsList = boardData.dataList[0];
    return tmsList.length > 0 ? tmsList[tmsList.length - 1] : null;
  }, [boardData]);

  const handleLogout = () => {
    if (confirm("로그아웃 하시겠습니까?")) {
      localStorage.clear();
      router.push("/");
    }
  };

  useEffect(() => {
    const savedRole = localStorage.getItem('userRole');
    if (!savedRole) { router.replace("/"); return; }
    setUserData({
      userNo: Number(localStorage.getItem('userNo')),
      userId: localStorage.getItem('userId') || "",
      userName: localStorage.getItem('userName') || "사용자",
      userRole: savedRole,
      userEmail: localStorage.getItem('userEmail') || "" 
    });
    setIsAuthChecked(true);
  }, [router]);

  if (!isAuthChecked) return <div className={isDarkMode ? "min-h-screen bg-slate-950" : "min-h-screen bg-[#f0f4f8]"} />;

  return (
    // 최상위 컨테이너에 dark 클래스 조건부 부여
    <div className={`min-h-screen lg:h-screen p-5 font-sans flex flex-col lg:overflow-hidden transition-colors duration-500 ${
      isDarkMode ? "dark bg-[#0a0f1d] text-slate-200" : "bg-[#f0f5fa] text-slate-800"
    }`}>
      
      {/* HEADER */}
      <header className={`flex justify-between items-center mb-6 border-b pb-5 h-14 flex-none relative transition-colors ${
        isDarkMode ? "border-white/5" : "border-blue-100"
      }`}>
        <div className="z-10">
          <h1 className={`text-xl font-black flex items-center gap-2.5 ${isDarkMode ? "text-white" : "text-slate-900"}`}>
            <span className={`w-3 h-3 rounded-full transition-all duration-500 shadow-lg ${isSystemOk ? "bg-blue-500 shadow-blue-500/80" : "bg-red-500 shadow-red-500/80"}`}></span>
            Smart WWTP <span className={isDarkMode ? "text-slate-400 font-light" : "text-blue-500 font-light"}>Monitoring</span>
          </h1>
        </div>

        {/* 중앙 상태 표시 및 날씨 */}
        <div className="hidden md:block absolute left-1/2 transform -translate-x-1/2">
          <div className="flex items-center gap-4">
            <div className={`px-4 py-1.5 rounded-xl border flex items-center gap-3 transition-all duration-500 ${
              isDarkMode ? "bg-emerald-500/5 border-emerald-500/20" : "bg-white/80 border-blue-200 shadow-sm"
            }`}>
              <span className={`text-xs font-bold ${isDarkMode ? "text-slate-400" : "text-blue-400"}`}>시스템:</span>
              <span className={`text-sm font-black ${isSystemOk ? "text-emerald-500" : "text-red-500"}`}>{isSystemOk ? "정상" : "점검필요"}</span>
              <span className="relative flex h-2 w-2">
                <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${isSystemOk ? "bg-emerald-400" : "bg-red-500"}`}></span>
                <span className={`relative inline-flex rounded-full h-2 w-2 ${isSystemOk ? "bg-emerald-500" : "bg-red-600"}`}></span>
              </span>
            </div>
            {latestWeather && (
              <div className={`px-4 py-1.5 rounded-xl border flex items-center gap-4 transition-all ${
                isDarkMode ? "border-white/10 bg-white/5 text-slate-100" : "border-blue-200 bg-white/80 text-blue-900 shadow-sm"
              }`}>
                <span className="text-sm font-bold">🌡️ {latestWeather.TA}°C</span>
                <span className="text-sm font-bold">💧 {latestWeather.RN_15m > 0 ? `${latestWeather.RN_15m}mm` : "강우없음"}</span>
              </div>
            )}
          </div>
        </div>

        {/* 우측 섹션 */}
        <div className="relative flex items-center gap-4 z-10">
          {/* 테마 전환 버튼 (관리자님 성함 왼쪽) */}
          <button 
            onClick={toggleTheme}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-xl border font-bold text-xs transition-all ${
              isDarkMode 
                ? "bg-white/5 border-white/10 text-yellow-400 hover:bg-white/10" 
                : "bg-white border-blue-200 text-indigo-600 hover:bg-blue-50 shadow-sm"
            }`}
          >
            <span>{isDarkMode ? "☀️ Light" : "🌙 Dark"}</span>
          </button>

          <div 
            onClick={() => setIsProfileOpen(!isProfileOpen)} 
            className={`flex items-center gap-3 px-3 py-1.5 rounded-2xl transition-all cursor-pointer border border-transparent group ${
              isDarkMode ? "hover:bg-white/5 hover:border-white/10" : "hover:bg-white/50 hover:border-blue-100"
            }`}
          >
            <div className="text-right hidden sm:block">
              <p className={`text-sm font-bold transition-colors ${isDarkMode ? "text-slate-100 group-hover:text-white" : "text-slate-800"}`}>
                {userData.userName}님
              </p>
              <p className="text-[10px] text-blue-500 font-bold uppercase tracking-widest">
                {userData.userRole.replace("ROLE_", "")}
              </p>
            </div>
            <div className="w-9 h-9 bg-linear-to-tr from-blue-600 to-indigo-600 rounded-xl flex items-center justify-center text-white font-black shadow-lg transform group-hover:rotate-3 transition-transform">
              {userData.userName.substring(0, 1)}
            </div>
          </div>

          <button onClick={handleLogout} className={`px-4 py-1.5 rounded-xl border transition-all text-xs font-bold ${
            isDarkMode 
              ? "bg-red-500/5 border-red-500/10 text-slate-400 hover:bg-red-500/20 hover:text-red-400" 
              : "bg-white border-red-100 text-red-500 hover:bg-red-50 shadow-sm"
          }`}>
            LOGOUT
          </button>

          <AnimatePresence>
            {isProfileOpen && (
              <motion.div 
                initial={{ opacity: 0, y: 10, scale: 0.95 }} 
                animate={{ opacity: 1, y: 0, scale: 1 }} 
                exit={{ opacity: 0, y: 10, scale: 0.95 }} 
                className={`absolute right-0 top-full mt-4 w-64 backdrop-blur-xl rounded-2xl z-50 overflow-hidden border shadow-2xl ${
                  isDarkMode ? "bg-slate-900/95 border-white/10" : "bg-white/95 border-blue-100"
                }`}
              >
                <div className="flex flex-col text-sm p-3 gap-1">
                  <button onClick={() => { setIsEditModalOpen(true); setIsProfileOpen(false); }} className={`flex items-center gap-3 px-4 py-3 rounded-xl transition-all font-medium text-left ${
                    isDarkMode ? "hover:bg-white/5 text-slate-300" : "hover:bg-blue-50 text-slate-700"
                  }`}>
                    <span className="text-lg">👤</span> 개인정보 수정
                  </button>
                  {userData.userRole === "ROLE_ADMIN" && (
                    <button onClick={() => { router.push("/admin/member"); setIsProfileOpen(false); }} className={`flex items-center gap-3 px-4 py-3 rounded-xl transition-all font-medium text-left ${
                      isDarkMode ? "hover:bg-white/5 text-slate-300" : "hover:bg-blue-50 text-slate-700"
                    }`}>
                      <span className="text-lg">⚙️</span> 이용자 관리
                    </button>
                  )}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </header>

      {/* MAIN CONTENT - 하위 컴포넌트에 isDarkMode 전달 */}
      <main className="flex-1 grid grid-cols-10 gap-6 lg:overflow-hidden min-h-0">
        <section className="col-span-12 lg:col-span-6 flex flex-col gap-6 min-h-0">
          <div className={`flex-1 border rounded-4xl overflow-hidden transition-all duration-500 ${
            isDarkMode ? "bg-white/2 border-white/5 shadow-inner" : "bg-white/70 border-white shadow-xl backdrop-blur-md"
          }`}>
            <Row1Charts isDarkMode={isDarkMode} />
          </div>
        </section>

        <section className="col-span-12 lg:col-span-4 flex flex-col gap-6 min-h-0">
          <div className={`flex-none border rounded-4xl overflow-hidden transition-all duration-500 ${
            isDarkMode ? "bg-white/2 border-white/5" : "bg-white/70 border-white shadow-lg"
          }`}>
            <Row2RiskDetail isDarkMode={isDarkMode} latestWeather={latestWeather} />
          </div>
          <div className={`flex-none border rounded-4xl overflow-hidden transition-all duration-500 ${
            isDarkMode ? "bg-white/2 border-white/5" : "bg-white/70 border-white shadow-lg"
          }`}>
            <Row3Alerts latestValues={latestTms} latestWeather={latestWeather} isDarkMode={isDarkMode} />
          </div>
          <div className={`border rounded-4xl overflow-hidden h-28 transition-all duration-500 ${
            isDarkMode ? "bg-white/2 border-white/5" : "bg-white/70 border-white shadow-lg"
          }`}>
            <Row4ActionPanel isDarkMode={isDarkMode} />
          </div>
        </section>
      </main>

      <EditProfileModal 
        isOpen={isEditModalOpen} 
        onClose={() => setIsEditModalOpen(false)} 
        currentUser={{ 
          userNo: userData.userNo, id: userData.userId, name: userData.userName, 
          role: userData.userRole, email: userData.userEmail 
        }} 
        onUpdateSuccess={(newId, newName, newEmail) => {
          setUserData(prev => ({ ...prev, userId: newId, userName: newName, userEmail: newEmail || prev.userEmail }));
          localStorage.setItem('userId', newId);
          localStorage.setItem('userName', newName);
          if (newEmail) localStorage.setItem('userEmail', newEmail);
        }}
      />
    </div>
  );
}