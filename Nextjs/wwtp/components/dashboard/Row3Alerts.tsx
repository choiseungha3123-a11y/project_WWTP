"use client";

import { motion } from "framer-motion";
import { AlertTriangle, Activity, Droplets, DatabaseZap, PencilLine } from "lucide-react";

export default function Row3Alerts() {
  const alerts = [
    { id: 1, title: "유입유량 기준초과", status: "danger", icon: <Activity className="w-4 h-4" /> },
    { id: 2, title: "TOC/TN/TP 안전 기준 근접 및 초과", status: "danger", icon: <AlertTriangle className="w-4 h-4" /> },
    { id: 3, title: "센서 이상 / 데이터 결측", status: "normal", icon: <DatabaseZap className="w-4 h-4" /> },
  ];

  // 조치하기 버튼 클릭 핸들러
  const handleActionClick = (title: string) => {
    // 커스텀 이벤트 생성 및 발송
    const event = new CustomEvent("setMemoInput", { 
      detail: { text: `[${title}] 조치 사항: ` } 
    });
    window.dispatchEvent(event);
  };

  return (
    <div className="bg-slate-800/40 p-4 rounded-3xl border border-white/10 h-full flex flex-col">
      <div className="flex justify-between items-center mb-3">
        <h3 className="text-sm font-bold text-slate-400 uppercase tracking-widest">Event Detection</h3>
        <span className="text-[9px] bg-red-500/20 text-red-400 px-2 py-0.5 rounded-full animate-pulse font-bold">LIVE</span>
      </div>
      
      <div className="space-y-2 flex-1 overflow-y-auto pr-1 custom-scrollbar">
        {alerts.map((alert, index) => (
          <motion.div
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.1 }}
            key={alert.id}
            className="flex items-center justify-between p-2.5 rounded-xl bg-slate-900/50 border border-white/5 hover:border-white/20 transition-all group/item"
          >
            <div className="flex items-center gap-3">
              <div className={`${
                alert.status === 'danger' ? 'text-red-500' : 
                alert.status === 'warning' ? 'text-orange-500' : 'text-blue-500'
              }`}>
                {alert.icon}
              </div>
              <span className="text-[13px] font-medium text-slate-200">{alert.id}. {alert.title}</span>
            </div>
            
            <div className="flex items-center gap-3">
              {/* 조치하기 버튼: 평소엔 숨겨져 있다가 호버 시 나타남 (모바일 대응 위해 opacity-0 sm:group-hover:opacity-100) */}
              <button 
                onClick={() => handleActionClick(alert.title)}
                className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-blue-600/20 hover:bg-blue-600 text-blue-400 hover:text-white text-[11px] font-bold transition-all border border-blue-500/30 opacity-0 group-hover/item:opacity-100"
              >
                <PencilLine className="w-3 h-3" />
                조치하기
              </button>

              <div className={`w-1.5 h-1.5 rounded-full shrink-0 ${
                alert.status === 'danger' ? 'bg-red-500 animate-ping' : 
                alert.status === 'warning' ? 'bg-orange-500' : 'bg-emerald-500'
              }`} />
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
}