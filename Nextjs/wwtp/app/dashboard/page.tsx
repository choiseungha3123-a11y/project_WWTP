"use client";

import { motion } from "framer-motion";

export default function DashboardPage() {
  // 샘플 데이터 (나중에 Spring Boot API에서 가져올 부분)
  const stats = [
    { name: "현재 유입 유량", value: "1,240 m³/h", status: "정상", color: "text-blue-400" },
    { name: "평균 pH 농도", value: "7.2 pH", status: "안정", color: "text-green-400" },
    { name: "BOD 유입 부하", value: "185 mg/L", status: "주의", color: "text-yellow-400" },
    { name: "송풍기 가동률", value: "85%", status: "운영중", color: "text-purple-400" },
  ];

  return (
    <div className="min-h-screen bg-slate-900 text-white p-8">
      {/* 상단 헤더 */}
      <header className="flex justify-between items-center mb-10 border-b border-white/10 pb-6">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-blue-400">Smart WWTP Dashboard</h1>
          <p className="text-slate-400 mt-1">실시간 하수처리 공정 모니터링 시스템</p>
        </div>
        <div className="flex items-center gap-4">
          <span className="bg-blue-500/20 text-blue-400 px-4 py-1 rounded-full text-sm border border-blue-500/30">
            관리자 모드
          </span>
          <button className="text-sm text-slate-400 hover:text-white transition-colors">로그아웃</button>
        </div>
      </header>

      {/* 요약 카드 섹션 */}
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

      {/* 메인 관제 레이아웃 */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* 공정 흐름도 요약 (왼쪽/중앙) */}
        <div className="lg:col-span-2 bg-slate-800/30 rounded-3xl border border-white/5 p-8 h-[400px] flex items-center justify-center relative overflow-hidden">
          <div className="absolute inset-0 opacity-20 pointer-events-none">
            {/* 여기에 배경으로 아까 그 이미지를 흐릿하게 넣을 수도 있습니다 */}
          </div>
          <p className="text-slate-500 italic text-lg">공정별 실시간 데이터 시각화 차트 영역</p>
        </div>

        {/* 실시간 알림창 (오른쪽) */}
        <div className="bg-slate-800/80 rounded-3xl border border-white/5 p-6 flex flex-col">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <span className="w-2 h-2 bg-red-500 rounded-full animate-ping" />
            최근 시스템 알림
          </h3>
          <div className="space-y-4 overflow-y-auto max-h-[300px]">
            {[1, 2, 3].map((i) => (
              <div key={i} className="p-3 bg-white/5 rounded-lg border-l-4 border-yellow-500 text-sm">
                <p className="font-medium text-yellow-500">주의: 유입 유량 급증</p>
                <p className="text-slate-400 text-xs">10분 전 - 제 1침전지 유입부</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}