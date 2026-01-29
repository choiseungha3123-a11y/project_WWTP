"use client";

import { useState, useEffect } from "react";
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

// 1. 개별 지표 카드 컴포넌트
const MetricCard = ({ 
  title, 
  value, 
  unit, 
  color, 
  onClick 
}: { 
  title: string; 
  value: string | number; 
  unit: string; 
  color: string; 
  onClick: () => void; 
}) => (
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
        <span className="text-4xl font-bold tracking-tight text-white">
          {value}
        </span>
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
  const [isAddModalOpen, setIsAddModalOpen] = useState(false);
  const [isEditModalOpen, setIsEditModalOpen] = useState(false);
  const [isAuthChecked, setIsAuthChecked] = useState(false);
  const [userNo, setUserNo] = useState<number>(0);
  const [userRole, setUserRole] = useState("");
  const [userId, setUserId] = useState("");
  const [userName, setUserName] = useState("");
  const [isProfileOpen, setIsProfileOpen] = useState(false);
  const [processData, setProcessData] = useState<ProcessDataItem[]>([]);

  useEffect(() => {
    const savedRole = localStorage.getItem('userRole');
    const savedId = localStorage.getItem('userId');
    const savedName = localStorage.getItem('userName');
    const savedNo = localStorage.getItem('userNo');
    
    if (savedNo) setUserNo(Number(savedNo));
    if (savedId) setUserId(savedId);
    
    if (!savedRole) {
      alert("로그인이 필요합니다.");
      router.replace("/");
    } else {
      setUserRole(savedRole);
      setUserName(savedName || "username");
      setIsAuthChecked(true);
    }

    fetch("/data/process_data.json")
      .then(res => res.json())
      .then(json => setProcessData(json))
      .catch(err => console.error("데이터 로딩 실패:", err));
  }, [router]);

  const handleLogout = () => {
    if (confirm("로그아웃 하시겠습니까?")) {
      router.push("/"); 
    }
  };

  // 최신 데이터 1건 추출
  const latest = processData.length > 0 ? processData[processData.length - 1] : null;

  // 지표 설정값 정의
  const metrics = [
    { title: "TOC (총유기탄소)", key: "toc", color: "#60a5fa", unit: "mg/L" },
    { title: "pH (수소이온농도)", key: "ph", color: "#34d399", unit: "pH" },
    { title: "SS (부유물질)", key: "ss", color: "#fbbf24", unit: "mg/L" },
    { title: "FLUX (유량)", key: "flux", color: "#a78bfa", unit: "m³/h" },
    { title: "TN (총질소)", key: "tn", color: "#f87171", unit: "mg/L" },
    { title: "TP (총인)", key: "tp", color: "#22d3ee", unit: "mg/L" },
  ];

  if (!isAuthChecked) return <div className="min-h-screen bg-slate-900" />;

  return (
    <div className="min-h-screen bg-slate-900 text-white p-8 font-sans">
      {/* ---------------- 상단 헤더 ---------------- */}
      <header className="flex justify-between items-center mb-12 border-b border-white/10 pb-8">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-white flex items-center gap-3">
            <span className="text-blue-500">●</span> Smart WWTP Monitoring
          </h1>
          <p className="text-slate-400 mt-2 font-light">공정별 실시간 상태 요약</p>
        </div>

        <div className="relative flex items-center gap-6">
          <div 
            onClick={() => setIsProfileOpen(!isProfileOpen)}
            className="flex items-center gap-3 hover:bg-white/5 p-2 rounded-xl transition-all cursor-pointer border border-transparent hover:border-white/10"
          >
            <div className="text-right">
              <p className="text-sm font-bold text-white">{userName}님</p>
              <p className="text-[10px] text-slate-500 uppercase tracking-widest">{userRole.replace("ROLE_", "")}</p>
            </div>
            <div className="w-10 h-10 bg-linear-to-br from-blue-500 to-indigo-600 rounded-full flex items-center justify-center text-sm text-white font-bold shadow-lg">
              {userName.substring(0, 1)}
            </div>
          </div>
          
          {/* 드롭다운 메뉴 (기존과 동일) */}
          <AnimatePresence>
  {isProfileOpen && (
    <motion.div 
      initial={{ opacity: 0, y: 10, scale: 0.95 }} 
      animate={{ opacity: 1, y: 0, scale: 1 }} 
      exit={{ opacity: 0, y: 10, scale: 0.95 }}
      className={`
        absolute right-0 top-full mt-4 w-64 
        /* 1. 배경을 더 밝게 변경 */
        bg-slate-700 
        rounded-2xl z-50 overflow-hidden 
        /* 2. 테두리를 더 밝고 선명하게 (눈에 띄는 구분선) */
        border border-slate-500/50 
        /* 3. 우측 하단 그림자에 파란색을 살짝 섞어 대비 증폭 */
        shadow-[15px_20px_40px_rgba(0,0,0,0.7),5px_5px_15px_rgba(59,130,246,0.1)]
      `}
    >
      {/* 상단 섹션: 더 밝은 배경색으로 강조 */}
      <div className="p-6 flex flex-col items-center border-b border-slate-600 bg-slate-600/50">
        <div className="relative mb-3">
          {/* 아바타 테두리에 밝은 글로우 추가 */}
          <div className="absolute inset-0 bg-blue-400 blur-md rounded-full opacity-30"></div>
          <div className="relative w-16 h-16 bg-blue-500 rounded-full flex items-center justify-center text-white text-xl font-bold border-2 border-white/20 shadow-lg">
            {userName.substring(0, 1)}
          </div>
        </div>
        <p className="font-bold text-lg text-white leading-tight">{userName}</p>
        <p className="text-xs text-blue-300/80 mt-1 font-medium">{userId}</p>
      </div>

      {/* 메뉴 리스트: 호버 시 더 밝은 색상으로 반응 */}
      <div className="flex flex-col text-sm p-2 bg-slate-700">
        <button 
          onClick={() => { setIsEditModalOpen(true); setIsProfileOpen(false); }} 
          className="flex items-center gap-3 text-left px-4 py-3 rounded-xl hover:bg-slate-600 text-slate-100 transition-all"
        >
          <span className="text-lg">👤</span>
          <span className="font-medium">개인정보 수정</span>
        </button>
        
        {userRole === "ROLE_ADMIN" && (
          <button 
            onClick={() => { router.push("/admin/member"); setIsProfileOpen(false); }} 
            className="flex items-center gap-3 text-left px-4 py-3 rounded-xl hover:bg-slate-600 text-slate-100 transition-all"
          >
            <span className="text-lg">⚙️</span>
            <span className="font-medium">사원 관리</span>
          </button>
        )}
        
        <div className="h-px bg-slate-600 my-1 mx-2"></div>

        <button 
          onClick={handleLogout} 
          className="flex items-center gap-3 text-left px-4 py-3 rounded-xl hover:bg-red-500/20 text-red-300 transition-all"
        >
          <span className="text-lg">🚪</span>
          <span className="font-bold">로그아웃</span>
        </button>
      </div>
    </motion.div>
  )}
</AnimatePresence>
        </div>
      </header>

      {/* ---------------- 지표 카드 그리드 섹션 ---------------- */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
        {metrics.map((metric, index) => (
          <MetricCard
            key={metric.key}
            title={metric.title}
            // 최신 데이터가 있으면 해당 key값 출력, 없으면 대기중 표시
            value={latest ? (latest[metric.key as keyof ProcessDataItem]) : "..."}
            unit={metric.unit}
            color={metric.color}
            // 클릭 시 상세 페이지로 이동 (예: /dashboard/toc)
            onClick={() => router.push(`/dashboard/${metric.key}`)}
          />
        ))}
      </div>

      {/* ---------------- 모달 섹션 ---------------- */}
      <EditProfileModal 
        isOpen={isEditModalOpen} 
        onClose={() => setIsEditModalOpen(false)} 
        currentUser={{ userNo, id: userId, name: userName, role: userRole }}
        onUpdateSuccess={(newId, newName) => { setUserId(newId); setUserName(newName); }}
      />
      <AddMemberModal 
        isOpen={isAddModalOpen}
        onClose={() => setIsAddModalOpen(false)}
        onSuccess={() => console.log("회원 등록 완료")}
      />
    </div>
  );
}