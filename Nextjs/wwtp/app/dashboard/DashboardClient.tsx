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
interface WeatherData {
  SYS_TIME: string;
  TA: number;      // 기온
  RN_15m: number;  // 강우량
  HM: number;      // 습도
}

interface TmsRecord {
  SYS_TIME: string;
  TOC_VU: number;
  PH_VU: number;
  SS_VU: number;
  FLUX_VU: number;
  TN_VU: number;
  TP_VU: number;
}

interface FlowRecord {
  SYS_TIME: string;
  Q_in: number;
}

interface BoardViewResponse {
  success: boolean;
  dataList: [TmsRecord[], TmsRecord[], FlowRecord[], FlowRecord[], WeatherData[]]; 
}

interface HealthResponse {
  success: boolean;
  message?: string;
  checkTime?: string;
}

const fetcher = async (url: string) => {
  const token = typeof window !== "undefined" ? localStorage.getItem("accessToken") : null;
  const response = await fetch(url, {
    method: "GET",
    headers: { 
      "Content-Type": "application/json", 
      ...(token ? { "Authorization": `Bearer ${token}` } : {}) 
    },
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
  const [errorStartTime, setErrorStartTime] = useState<string | null>(null);

  // 데이터 페칭 (30초 간격 갱신)
  const { data: healthRaw, error: healthError } = useSWR<HealthResponse>(
    `${API_BASE_URL}/api/board/health`, fetcher, { refreshInterval: 30000 }
  );

  const { data: boardData } = useSWR<BoardViewResponse>(
    `${API_BASE_URL}/api/board/boardView`, fetcher, { refreshInterval: 30000 }
  );

  const isSystemOk = !!healthRaw && healthRaw.success === true && !healthError;

  // 최신 날씨 데이터 추출
  const latestWeather = useMemo(() => {
    if (!boardData?.success || !boardData.dataList[4]) return null;
    const weatherList = boardData.dataList[4];
    return weatherList.length > 0 ? weatherList[weatherList.length - 1] : null;
  }, [boardData]);

  // 최신 TMS 실측 데이터 추출 (Row3Alerts 전달용)
  const latestTms = useMemo(() => {
    if (!boardData?.success || !boardData.dataList[0]) return null;
    const tmsList = boardData.dataList[0];
    return tmsList.length > 0 ? tmsList[tmsList.length - 1] : null;
  }, [boardData]);

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

  if (!isAuthChecked) return <div className="min-h-screen bg-slate-950" />;

  return (
    <div className="min-h-screen lg:h-screen bg-[#0a0f1d] text-slate-200 p-5 font-sans flex flex-col lg:overflow-hidden">
      
      {/* HEADER */}
      <header className="flex justify-between items-center mb-6 border-b border-white/5 pb-5 h-14 flex-none relative">
        <div className="z-10">
          <h1 className="text-xl font-black text-white flex items-center gap-2.5">
            <span className={`w-3 h-3 rounded-full ${isSystemOk ? "bg-blue-500 shadow-blue-500/50" : "bg-red-500 shadow-red-500/50 shadow-lg"}`}></span>
            Smart WWTP <span className="text-slate-400 font-light">Monitoring</span>
          </h1>
        </div>

        {/* 중앙 상태 표시 (System + Weather) */}
        <div className="hidden md:block absolute left-1/2 transform -translate-x-1/2">
          <div className="flex items-center gap-4">
            <div className={`px-4 py-1.5 rounded-xl border flex items-center gap-3 ${isSystemOk ? "bg-emerald-500/5 border-emerald-500/20" : "bg-red-500/10 border-red-500/40"}`}>
              <span className="text-xs font-bold text-slate-400">시스템:</span>
              <span className={`text-sm font-black ${isSystemOk ? "text-emerald-400" : "text-red-400"}`}>{isSystemOk ? "정상" : "점검필요"}</span>
            </div>
            {latestWeather && (
              <div className="px-4 py-1.5 rounded-xl border border-white/10 bg-white/5 flex items-center gap-4">
                <span className="text-sm font-bold text-slate-100">🌡️ {latestWeather.TA}°C</span>
                <span className="text-sm font-bold text-slate-100">💧 {latestWeather.RN_15m > 0 ? `${latestWeather.RN_15m}mm` : "강우없음"}</span>
              </div>
            )}
          </div>
        </div>

        <div className="flex items-center gap-4 z-10">
          <div onClick={() => setIsProfileOpen(!isProfileOpen)} className="cursor-pointer flex items-center gap-3 group">
            <div className="text-right hidden sm:block">
              <p className="text-sm font-bold text-white">{userData.userName}님</p>
              <p className="text-[10px] text-blue-500 font-bold uppercase">{userData.userRole.replace("ROLE_", "")}</p>
            </div>
            <div className="w-9 h-9 bg-blue-600 rounded-xl flex items-center justify-center font-black shadow-lg group-hover:scale-105 transition-transform">
              {userData.userName.substring(0, 1)}
            </div>
          </div>
        </div>
      </header>

      {/* MAIN CONTENT */}
      <main className="flex-1 grid grid-cols-10 gap-6 lg:overflow-hidden min-h-0">
        <section className="col-span-12 lg:col-span-6 flex flex-col gap-6 min-h-0">
          <div className="flex-1 bg-white/2 border border-white/5 rounded-4xl overflow-hidden shadow-inner">
            <Row1Charts />
          </div>
        </section>

        <section className="col-span-12 lg:col-span-4 flex flex-col gap-6 min-h-0">
          <div className="flex-none bg-white/2 border border-white/5 rounded-4xl">
            <Row2RiskDetail />
          </div>
          {/* Row3Alerts에 실시간 데이터 전달 */}
          <div className="flex-none bg-white/2 border border-white/5 rounded-4xl overflow-hidden h-[240px]">
            <Row3Alerts latestValues={latestTms} latestWeather={latestWeather} />
          </div>
          <div className="flex-1 bg-white/2 border border-white/5 rounded-4xl overflow-hidden">
            <Row4ActionPanel />
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
        }}
      />
    </div>
  );
}