"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import EditProfileModal from "../components/EditProfileModal";
import AddMemberModal from "../components/AddMemberModal"

export default function DashboardPage() {
  const [isAddModalOpen, setIsAddModalOpen] = useState(false);
  const [isEditModalOpen, setIsEditModalOpen] = useState(false);
  const router = useRouter();
  const [isAuthChecked, setIsAuthChecked] = useState(false); // 인증 체크 완료 여부
  const [userNo, setUserNo] = useState<number>(0);
  const [userRole, setUserRole] = useState("");
  const [userId, setUserId] = useState("");
  const [userName, setUserName] = useState("");
  const [isProfileOpen, setIsProfileOpen] = useState(false);

  useEffect(() => {
    const savedRole = localStorage.getItem('userRole');
    const savedId = localStorage.getItem('userId');
    const savedName = localStorage.getItem('userName');
    const savedNo = localStorage.getItem('userNo');
    
    if (savedNo) setUserNo(Number(savedNo));
    if (savedId) setUserId(savedId);
    if (savedName) setUserName(savedName);
    if (!savedRole) {
      console.error("인증 실패: userRole 없음");
      alert("로그인이 필요합니다.");
      router.replace("/");
    } else {
      setUserRole(savedRole);
      setUserName(savedName || "username");
      setIsAuthChecked(true);
    }
  }, [router]);

  if (!isAuthChecked) {
    return <div className="min-h-screen bg-slate-900" />;
  }

  const handleLogout = () => {
    if (confirm("로그아웃 하시겠습니까?")) {
      router.push("/"); 
    }
  };
  
  // 샘플 데이터 (나중에 Spring Boot API에서 가져올 부분)
  const stats = [
    { name: "현재 유입 유량", value: "1,240 m³/h", status: "정상", color: "text-blue-400" },
    { name: "평균 pH 농도", value: "7.2 pH", status: "안정", color: "text-green-400" },
    { name: "BOD 유입 부하", value: "185 mg/L", status: "주의", color: "text-yellow-400" },
    { name: "송풍기 가동률", value: "85%", status: "운영중", color: "text-purple-400" },
  ];

  return (
    <div className="min-h-screen bg-slate-900 text-white p-8">
      <header className="flex justify-between items-center mb-10 border-b border-white/10 pb-6">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-blue-400">Smart WWTP Dashboard</h1>
          <p className="text-slate-400 mt-1">실시간 하수처리 공정 모니터링 시스템</p>
        </div>

        <div className="relative flex items-center gap-6">
          <div className="flex items-center gap-4">
            <div 
              onClick={() => setIsProfileOpen(!isProfileOpen)}
              className="flex items-center gap-3 hover:bg-white/5 p-2 rounded-lg transition-all cursor-pointer"
            >
            <span className="text-sm font-medium text-white">{userName}님</span>
            <div className="w-8 h-8 bg-slate-700 rounded-full overflow-hidden border border-white/20">
            <div className="w-full h-full bg-blue-500 flex items-center justify-center text-xs text-white font-bold">
                AD
              </div>
            </div>
          </div>
          <button 
            onClick={(e) => {
              e.stopPropagation(); 
              handleLogout();
            }}
            className="text-sm text-slate-400 hover:text-white hover:font-bold transition-all p-1"
          >
            로그아웃
          </button>

          <AnimatePresence>
             {isProfileOpen && (
              <motion.div 
                 initial={{ opacity: 0, y: 10 }}
                 animate={{ opacity: 1, y: 0 }}
                 exit={{ opacity: 0, y: 10 }}
                 className="absolute right-0 top-full mt-2 w-64 bg-white text-slate-800 rounded-xl shadow-2xl z-50 overflow-hidden border border-slate-200"
              >
                <div className="flex border-b text-center text-sm">
                  <div className="flex-1 py-3 bg-slate-50 font-bold border-r text-blue-600">My</div>
                  <div className="flex-1 py-3 hover:bg-gray-100 cursor-pointer border-r text-gray-400">알림</div>
                  <div 
                    className="px-4 py-3 hover:bg-gray-100 cursor-pointer text-gray-400"
                    onClick={() => setIsProfileOpen(false)}
                    >✕</div>
                </div>

                <div className="p-6 flex flex-col items-center border-b">
                  <div className="w-16 h-16 bg-slate-200 rounded-full mb-3 overflow-hidden border border-gray-100">
                      <div className="w-full h-full bg-blue-500 flex items-center justify-center text-white text-xl">CP</div>
                  </div>
                  <p className="font-bold text-lg">{userName}</p>
                  <select className="mt-2 text-xs border rounded px-2 py-1 outline-none">
                    <option>한국어 (ko)</option>
                  </select>
                </div>
                  
                <div className="flex flex-col text-sm">
                  <button 
                    onClick={() => {
                                    setIsEditModalOpen(true);
                                    setIsProfileOpen(false); 
                                    }}
                    className="text-left px-6 py-4 hover:bg-blue-50 transition-colors border-b">
                      개인정보 수정
                  </button>
                    
                    {userRole === "ROLE_ADMIN" && (
                      <button 
                      onClick={() => {
                        router.push("/admin/member");
                        setIsProfileOpen(false);
                      }}
                      className="text-left px-6 py-4 hover:bg-blue-50 transition-colors border-b">
                        사원 관리
                      </button>
                    )}
                    
                    <button 
                      onClick={handleLogout}
                      className="text-left px-6 py-4 hover:bg-red-50 text-red-500 transition-colors"
                    >로그아웃</button>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
      </div>
    </div>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-10">
        {stats.map((stat, index) => (
          <motion.div
            key={index}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            className="bg-slate-800/50 p-6 rounded-2xl border border-white/5 backdrop-blur-sm hover:border-blue-500/50 transition-all shadow-xl"
          >
            <p className="text-sm text-slate-400 mb-2">{stat.name}</p>
            <h3 className={`text-2xl font-bold ${stat.color} mb-1`}>{stat.value}</h3>
            <p className="text-xs opacity-60">현재 상태: {stat.status}</p>
          </motion.div>
        ))}
      </div>

      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
      
        <div className="lg:col-span-2 bg-slate-800/30 rounded-3xl border border-white/5 p-8 h-100 flex items-center justify-center relative overflow-hidden">
          <div className="absolute inset-0 opacity-20 pointer-events-none">
      
          </div>
          <p className="text-slate-500 italic text-lg">공정별 실시간 데이터 시각화 차트 영역</p>
        </div>

      
        <div className="bg-slate-800/80 rounded-3xl border border-white/5 p-6 flex flex-col">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <span className="w-2 h-2 bg-red-500 rounded-full animate-ping" />
            최근 시스템 알림
          </h3>
          <div className="space-y-4 overflow-y-auto max-h-75">
            {[1, 2, 3].map((i) => (
              <div key={i} className="p-3 bg-white/5 rounded-lg border-l-4 border-yellow-500 text-sm">
                <p className="font-medium text-yellow-500">주의: 유입 유량 급증</p>
                <p className="text-slate-400 text-xs">10분 전 - 제 1침전지 유입부</p>
              </div>
            ))}
          </div>
        </div>
      </div>
      <EditProfileModal 
        isOpen={isEditModalOpen} 
        onClose={() => setIsEditModalOpen(false)} 
        currentUser={{ userNo: userNo, id: userId, name: userName, role: userRole }}
        onUpdateSuccess={(newId, newName) => {
          setUserId(newId);
          setUserName(newName);
        }}
      />

      <AddMemberModal 
        isOpen={isAddModalOpen}
        onClose={() => setIsAddModalOpen(false)}
        onSuccess={() => {
          console.log("회원 등록 완료");
        }}
      />
    </div>
  );
}