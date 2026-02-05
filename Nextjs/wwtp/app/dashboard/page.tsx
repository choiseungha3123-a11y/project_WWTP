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

 const [userData, setUserData] = useState(() => {
    if (typeof window !== "undefined") {
      const savedRole = localStorage.getItem('userRole');
      if (savedRole) {
        return {
          userNo: Number(localStorage.getItem('userNo')),
          userId: localStorage.getItem('userId') || "",
          userName: localStorage.getItem('userName') || "사용자",
          userRole: savedRole,
          isLoaded: true 
        };
      }
    }
    return { userNo: 0, userId: "", userName: "", userRole: "", isLoaded: false };
  });

  const [isProfileOpen, setIsProfileOpen] = useState(false);
  const [isEditModalOpen, setIsEditModalOpen] = useState(false);

  
  useEffect(() => {
    if (typeof window === "undefined") return;

    if (!userData.userRole && !localStorage.getItem('userRole')) {
      router.replace("/");
    }
  }, [userData.userRole, router]);

  if (!userData.isLoaded) {
    return <div className="min-h-screen bg-slate-950" />;
  }

  const handleLogout = () => {
    if (confirm("로그아웃 하시겠습니까?")) {
      localStorage.clear();
      router.push("/");
    }
  };

  return (
    <div className="min-h-screen lg:h-screen bg-slate-950 text-white p-4 font-sans flex flex-col lg:overflow-hidden">
      
      {/* --- 상단 헤더 --- */}
      <header className="flex justify-between items-center mb-4 border-b border-white/5 pb-4 h-16 flex-none">
        <div>
          <h1 className="text-xl font-black text-white flex items-center gap-3">
            <span className="text-blue-500">●</span> Smart WWTP Monitoring
          </h1>
          <p className="text-slate-500 text-[10px] uppercase tracking-widest font-medium">Integrated Operation Dashboard</p>
        </div>

        <div className="relative flex items-center gap-4">
          <div 
            onClick={() => setIsProfileOpen(!isProfileOpen)} 
            className="flex items-center gap-3 hover:bg-white/5 p-2 rounded-xl transition-all cursor-pointer border border-transparent hover:border-white/10"
          >
            <div className="text-right hidden sm:block"> 
              <p className="text-sm font-bold text-white">{userData.userName}님</p>
              <p className="text-[10px] text-blue-400 font-medium uppercase tracking-tighter">{userData.userRole.replace("ROLE_", "")}</p>
            </div>
            <div className="w-10 h-10 bg-linear-to-br from-blue-600 to-indigo-700 rounded-full flex items-center justify-center text-sm font-bold shadow-lg shadow-blue-900/20">
              {userData.userName.substring(0, 1)}
            </div>
          </div>

          <button onClick={handleLogout} className="px-3 py-1.5 rounded-lg bg-white/5 hover:bg-red-500/10 hover:text-red-400 border border-white/10 transition-all text-[11px] font-medium text-slate-400">
            로그아웃
          </button>
          
          <AnimatePresence>
            {isProfileOpen && (
              <motion.div 
                initial={{ opacity: 0, y: 10 }} 
                animate={{ opacity: 1, y: 0 }} 
                exit={{ opacity: 0, y: 10 }} 
                className="absolute right-0 top-full mt-4 w-64 bg-slate-800 rounded-2xl z-50 overflow-hidden border border-white/10 shadow-2xl"
              >
                <div className="p-6 flex flex-col items-center border-b border-white/5 bg-white/5">
                  <div className="w-16 h-16 bg-blue-600 rounded-full flex items-center justify-center text-white text-xl font-bold shadow-lg">{userData.userName.substring(0, 1)}</div>
                  <p className="font-bold text-lg text-white mt-3">{userData.userName}</p>
                  <p className="text-xs text-slate-400">{userData.userId}</p>
                </div>
                <div className="flex flex-col text-sm p-2">
                  <button onClick={() => { setIsEditModalOpen(true); setIsProfileOpen(false); }} className="flex items-center gap-3 px-4 py-3 rounded-xl hover:bg-white/5 transition-all text-slate-300">👤 개인정보 수정</button>
                  {userData.userRole === "ROLE_ADMIN" && (
                    <button onClick={() => { router.push("/admin/member"); setIsProfileOpen(false); }} className="flex items-center gap-3 px-4 py-3 rounded-xl hover:bg-white/5 transition-all text-slate-300">⚙️ 사원 관리</button>
                  )}
                  <button onClick={handleLogout} className="flex items-center gap-3 px-4 py-3 rounded-xl hover:bg-red-500/10 text-red-400 transition-all font-bold">🚪 로그아웃</button>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </header>

      <div className="flex-1 grid grid-cols-12 gap-4 lg:overflow-hidden min-h-0">
        <div className="col-span-12 lg:col-span-7 flex flex-col gap-4 min-h-0">
          <div className="flex-none">
            <Row1Status />
          </div>
          <div className="flex-none lg:flex-1 lg:min-h-0">
            <Row2Alerts />
          </div>
          <div className="flex-none lg:flex-1 lg:min-h-0">
            <Row3Charts />
          </div>
        </div>
        <div className="col-span-12 lg:col-span-5 flex flex-col gap-4 min-h-0">
          <div className="flex-none lg:flex-[0.55] lg:min-h-0">
            <Row4RiskDetail />
          </div>
          <div className="flex-none lg:flex-[0.45] lg:min-h-0">
            <Row5ActionPanel />
          </div>
        </div>
      </div>

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