"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";

// 분리한 컴포넌트들
import Row1Status from "@/components/dashboard/Row1Status";
import Row2Alerts from "@/components/dashboard/Row2Alerts";
import Row3Charts from "@/components/dashboard/Row3Charts";
import Row4RiskDetail from "@/components/dashboard/Row4RiskDetail";
import Row5ActionPanel from "@/components/dashboard/Row5ActionPanel";

import EditProfileModal from "../options/EditProfileModal";

export default function DashboardPage() {
  const router = useRouter();

  const [isAuthChecked, setIsAuthChecked] = useState(false);
  const [userData, setUserData] = useState({
    userNo: 0,
    userId: "",
    userName: "",
    userRole: ""
  });
  const [isProfileOpen, setIsProfileOpen] = useState(false);
  const [isEditModalOpen, setIsEditModalOpen] = useState(false);
  
  // [추가] 오늘 날짜 상태 관리
  const [todayDate, setTodayDate] = useState("");

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
      userRole: savedRole
    });
    
    // [추가] 날짜 포맷팅 (YYYY년 MM월 DD일 요일)
    const now = new Date();
    const formattedDate = now.toLocaleDateString('ko-KR', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      weekday: 'long'
    });
    setTodayDate(formattedDate);

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
    /* [개선 1] 배경색을 더 깊은 slate-950으로 고정하고 padding을 조절하여 여백의 미 확보 */
    <div className="min-h-screen lg:h-screen bg-[#0a0f1d] text-slate-200 p-5 font-sans flex flex-col lg:overflow-hidden selection:bg-blue-500/30">
      
      {/* --- 상단 헤더 --- */}
      <header className="flex justify-between items-center mb-6 border-b border-white/5 pb-5 h-14 flex-none relative">
        <div className="z-10">
          <h1 className="text-xl font-black text-white flex items-center gap-2.5 tracking-tight">
            <span className="w-3 h-3 rounded-full bg-blue-500 shadow-[0_0_12px_rgba(59,130,246,0.8)]"></span>
            Smart WWTP <span className="text-slate-400 font-light">Monitoring</span>
          </h1>
          <p className="text-slate-500 text-[9px] uppercase tracking-[0.2em] font-bold mt-1">Integrated Operation Center</p>
        </div>

        {/* [개선 3] 날짜 영역의 가독성 및 디자인 디테일 (유리질 효과 적용) */}
        <div className="hidden md:block absolute left-1/2 transform -translate-x-1/2 group">
          <div className="px-5 py-2 rounded-2xl bg-white/3 backdrop-blur-md border border-white/10 text-slate-300 text-xs font-semibold flex items-center gap-3 shadow-2xl transition-all hover:border-blue-500/30">
            <span className="text-blue-400 group-hover:scale-110 transition-transform">📅</span> {todayDate}
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
            {/* 프로필 이미지 그라데이션 및 그림자 강화 */}
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
                    <button 
                      onClick={() => { router.push("/admin/member"); setIsProfileOpen(false); }} 
                      className="flex items-center gap-3 px-4 py-3 rounded-xl hover:bg-white/5 transition-all text-slate-300 font-medium"
                    >
                      <span className="text-lg text-slate-500">⚙️</span> 사원 관리
                    </button>
                    <button 
                      onClick={() => { router.push("/admin/memo-history"); setIsProfileOpen(false); }} 
                      className="flex items-center gap-3 px-4 py-3 rounded-xl hover:bg-white/5 transition-all text-slate-300 font-medium"
                    >
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

      {/* 대시보드 메인 콘텐츠 */}
      {/* [개선 4] 그리드 간격을 gap-6으로 넓혀 컴포넌트 간 독립성 확보 */}
      <main className="flex-1 grid grid-cols-12 gap-6 lg:overflow-hidden min-h-0">
        
        {/* 좌측 영역 (주요 지표 및 알람) */}
        <section className="col-span-12 lg:col-span-7 flex flex-col gap-6 min-h-0">
          <div className="flex-none drop-shadow-sm">
            <Row1Status />
          </div>
          
          {/* [개선 5] 카드형 레이아웃의 통일감을 위해 각 로우를 감싸는 영역의 min-h 설정 최적화 */}
          <div className="flex-none lg:flex-1 lg:min-h-0 bg-white/2 border border-white/5 rounded-4xl overflow-hidden">
            <Row2Alerts />
          </div>
          
          <div className="flex-none lg:flex-1 lg:min-h-0 bg-white/2 border border-white/5 rounded-4xl overflow-hidden">
            <Row3Charts />
          </div>
        </section>

        {/* 우측 영역 (위험도 상세 및 조치 패널) */}
        <section className="col-span-12 lg:col-span-5 flex flex-col gap-6 min-h-0">
          {/* [개선 6] Flex 비율 조정: 위험도 상세(Row4)에 더 많은 공간 할당 */}
          <div className="flex-none lg:flex-[0.6] lg:min-h-0 bg-white/2 border border-white/5 rounded-4xl overflow-hidden shadow-inner">
            <Row4RiskDetail />
          </div>
          
          <div className="flex-none lg:flex-[0.4] lg:min-h-0 bg-white/2 border border-white/5 rounded-4xl overflow-hidden">
            <Row5ActionPanel />
          </div>
        </section>
      </main>

      <EditProfileModal 
        isOpen={isEditModalOpen} 
        onClose={() => setIsEditModalOpen(false)} 
        currentUser={{ 
          userNo: userData.userNo, 
          id: userData.userId, 
          name: userData.userName, 
          role: userData.userRole 
        }} 
        onUpdateSuccess={(newId, newName) => { 
          setUserData(prev => ({ ...prev, userId: newId, userName: newName })); 
        }} 
      />
    </div>
  );
}