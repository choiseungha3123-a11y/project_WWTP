"use client";

import { motion } from "framer-motion";
import { AlertTriangle, Activity, Droplets, DatabaseZap } from "lucide-react";

export default function Row3Alerts() {
  const alerts = [
    { id: 1, title: "유입유량 기준초과", status: "danger", icon: <Activity className="w-4 h-4" /> },
    { id: 2, title: "조정조 수위 기준초과", status: "warning", icon: <Droplets className="w-4 h-4" /> },
    { id: 3, title: "TOC/TN/TP 안전 기준 근접 및 초과", status: "danger", icon: <AlertTriangle className="w-4 h-4" /> },
    { id: 4, title: "센서 이상 / 데이터 결측", status: "normal", icon: <DatabaseZap className="w-4 h-4" /> },
  ];

  return (
    // p-5 -> p-4로 줄여 내부 여백 확보
    <div className="bg-slate-800/40 p-4 rounded-3xl border border-white/10 h-full flex flex-col">
      <div className="flex justify-between items-center mb-3">
        <h3 className="text-sm font-bold text-slate-400 uppercase tracking-widest">Event Detection</h3>
        <span className="text-[9px] bg-red-500/20 text-red-400 px-2 py-0.5 rounded-full animate-pulse font-bold">LIVE</span>
      </div>
      
      {/* space-y-3 -> space-y-2로 줄여 세로 공간 절약 */}
      <div className="space-y-2 flex-1 overflow-y-auto pr-1 custom-scrollbar">
        {alerts.map((alert, index) => (
          <motion.div
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.1 }}
            key={alert.id}
            // p-3 -> p-2.5로 축소
            className="flex items-center justify-between p-2.5 rounded-xl bg-slate-900/50 border border-white/5 hover:border-white/20 transition-all"
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
            <div className={`w-1.5 h-1.5 rounded-full ${
              alert.status === 'danger' ? 'bg-red-500 animate-ping' : 
              alert.status === 'warning' ? 'bg-orange-500' : 'bg-emerald-500'
            }`} />
          </motion.div>
        ))}
      </div>
    </div>
  );
}