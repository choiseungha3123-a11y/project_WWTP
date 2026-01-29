"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
// 차트 라이브러리 import
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import EditProfileModal from "../components/EditProfileModal";
import AddMemberModal from "../components/AddMemberModal";

// 1. JSON 데이터의 구조 정의 (TypeScript 오류 방지)
interface ProcessDataItem {
  time: string;
  toc: number;
  ph: number;
  ss: number;
  flux: number;
  tn: number;
  tp: number;
}

// 2. 공통 차트 컴포넌트 (반복되는 6개 그래프를 효율적으로 생성)
const MiniChart = ({ title, data, dataKey, color, unit }: { title: string, data: ProcessDataItem[], dataKey: string, color: string, unit: string }) => (
  <div className="bg-slate-800/50 p-4 rounded-2xl border border-white/5 shadow-xl h-64 hover:border-blue-500/30 transition-all flex flex-col">
    <div className="flex justify-between items-center mb-2">
      <h3 className="text-sm font-semibold text-slate-300">{title}</h3>
      <span className="text-[10px] font-bold px-2 py-0.5 rounded bg-white/5 text-slate-400">{unit}</span>
    </div>
    <div className="flex-1 w-full h-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#ffffff05" vertical={false} />
          <XAxis dataKey="time" hide />
          <YAxis 
            domain={['auto', 'auto']} 
            fontSize={10} 
            tick={{fill: '#64748b'}} 
            tickLine={false} 
            axisLine={false} 
          />
          <Tooltip 
            contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px', fontSize: '11px' }}
            itemStyle={{ color: '#fff' }}
          />
          <Line 
            type="monotone" 
            dataKey={dataKey} 
            stroke={color} 
            strokeWidth={2} 
            dot={false} 
            isAnimationActive={false} 
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  </div>
);

export default function DashboardPage() {
  const [isAddModalOpen, setIsAddModalOpen] = useState(false);
  const [isEditModalOpen, setIsEditModalOpen] = useState(false);
  const router = useRouter();
  const [isAuthChecked, setIsAuthChecked] = useState(false);
  const [userNo, setUserNo] = useState<number>(0);
  const [userRole, setUserRole] = useState("");
  const [userId, setUserId] = useState("");
  const [userName, setUserName] = useState("");
  const [isProfileOpen, setIsProfileOpen] = useState(false);

  // 실제 JSON 데이터를 저장할 상태
  const [processData, setProcessData] = useState<ProcessDataItem[]>([]);

  useEffect(() => {
    // 로컬 스토리지 인증 정보 가져오기
    const savedRole = localStorage.getItem('userRole');
    const savedId = localStorage.getItem('userId');
    const savedName = localStorage.getItem('userName');
    const savedNo = localStorage.getItem('userNo');
    
    if (savedNo) setUserNo(Number(savedNo));
    if (savedId) setUserId(savedId);
    if (savedName) setUserName(savedName);
    
    if (!savedRole) {
      alert("로그인이 필요합니다.");
      router.replace("/");
    } else {
      setUserRole(savedRole);
      setUserName(savedName || "username");
      setIsAuthChecked(true);
    }

    // JSON 데이터 호출
    fetch("/data/process_data.json")
      .then(res => res.json())
      .then(json => setProcessData(json))
      .catch(err => console.error("데이터 로딩 실패:", err));
  }, [router]);

  if (!isAuthChecked) {
    return <div className="min-h-screen bg-slate-900" />;
  }

  const handleLogout = () => {
    if (confirm("로그아웃 하시겠습니까?")) {
      router.push("/"); 
    }
  };
  
  // 6개 차트의 설정값
  const chartConfigs = [
    { title: "TOC (유기물 농도)", key: "toc", color: "#60a5fa", unit: "mg/L" },
    { title: "pH (수소이온 농도)", key: "ph", color: "#34d399", unit: "pH" },
    { title: "SS (부유물질)", key: "ss", color: "#fbbf24", unit: "mg/L" },
    { title: "FLUX (유입유량)", key: "flux", color: "#a78bfa", unit: "m³/h" },
    { title: "TN (총질소)", key: "tn", color: "#f87171", unit: "mg/L" },
    { title: "TP (총인)", key: "tp", color: "#22d3ee", unit: "mg/L" },
  ];

  // 최신 데이터 1건 추출 (상단 요약 카드용)
  const latest = processData.length > 0 ? processData[processData.length - 1] : ({} as Partial<ProcessDataItem>);

  // 상단 요약 카드 데이터 구성
  const stats = [
    { 
      name: "현재 유입 유량", 
      value: latest.flux ? `${latest.flux.toLocaleString()} m³/h` : "연결 중...", 
      status: "정상", 
      color: "text-blue-400" 
    },
    { 
      name: "현재 pH 농도", 
      value: latest.ph ? `${latest.ph} pH` : "연결 중...", 
      status: "안정", 
      color: "text-green-400" 
    },
    { 
      name: "현재 TOC 농도", 
      value: latest.toc ? `${latest.toc} mg/L` : "연결 중...", 
      status: (latest.toc ?? 0) > 4.3 ? "주의" : "정상", 
      color: (latest.toc ?? 0) > 4.3 ? "text-red-400" : "text-yellow-400" 
    },
    { 
      name: "현재 SS 농도", 
      value: latest.ss ? `${latest.ss} mg/L` : "연결 중...", 
      status: "운영중", 
      color: "text-purple-400" 
    },
  ];

  return (
    <div className="min-h-screen bg-slate-900 text-white p-8 font-sans">
      {/* ---------------- 상단 헤더 섹션 ---------------- */}
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
              <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center text-xs text-white font-bold border border-white/20">
                AD
              </div>
            </div>
            <button 
              onClick={(e) => { e.stopPropagation(); handleLogout(); }}
              className="text-sm text-slate-400 hover:text-white transition-all p-1"
            >
              로그아웃
            </button>
          </div>

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
                  <div className="px-4 py-3 hover:bg-gray-100 cursor-pointer text-gray-400" onClick={() => setIsProfileOpen(false)}>✕</div>
                </div>
                <div className="p-6 flex flex-col items-center border-b">
                  <div className="w-16 h-16 bg-blue-500 rounded-full mb-3 flex items-center justify-center text-white text-xl font-bold">CP</div>
                  <p className="font-bold text-lg">{userName}</p>
                </div>
                <div className="flex flex-col text-sm">
                  <button onClick={() => { setIsEditModalOpen(true); setIsProfileOpen(false); }} className="text-left px-6 py-4 hover:bg-blue-50 border-b transition-colors">
                    개인정보 수정
                  </button>
                  {userRole === "ROLE_ADMIN" && (
                    <button onClick={() => { router.push("/admin/member"); setIsProfileOpen(false); }} className="text-left px-6 py-4 hover:bg-blue-50 border-b transition-colors">
                      사원 관리
                    </button>
                  )}
                  <button onClick={handleLogout} className="text-left px-6 py-4 hover:bg-red-50 text-red-500 transition-colors">
                    로그아웃
                  </button>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </header>

      {/* ---------------- 상단 요약 카드 섹션 ---------------- */}
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

      {/* ---------------- 메인 차트 및 알림 섹션 ---------------- */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* 왼쪽: 6개 차트 그리드 (2열 3행 구성) */}
        <div className="lg:col-span-2 grid grid-cols-1 md:grid-cols-2 gap-4">
          {chartConfigs.map((cfg) => (
            <MiniChart 
              key={cfg.key}
              title={cfg.title}
              data={processData}
              dataKey={cfg.key}
              color={cfg.color}
              unit={cfg.unit}
            />
          ))}
        </div>

        {/* 오른쪽: 최근 알림 섹션 */}
        <div className="bg-slate-800/80 rounded-3xl border border-white/5 p-6 flex flex-col h-fit shadow-2xl">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <span className="w-2 h-2 bg-red-500 rounded-full animate-ping" />
            최근 시스템 알림
          </h3>
          <div className="space-y-4 overflow-y-auto max-h-125">
            {[1, 2, 3].map((i) => (
              <div key={i} className="p-4 bg-white/5 rounded-xl border-l-4 border-yellow-500 text-sm hover:bg-white/10 transition-colors">
                <p className="font-medium text-yellow-500">주의: 유입 유량 임계치 근접</p>
                <p className="text-slate-400 text-xs mt-1">방금 전 - 수처리 제1공정</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ---------------- 모달 섹션 ---------------- */}
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
        onSuccess={() => console.log("회원 등록 완료")}
      />
    </div>
  );
}