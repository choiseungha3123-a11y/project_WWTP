"use client";

import { motion } from "framer-motion";
import { AlertTriangle, Activity, DatabaseZap, PencilLine, CheckCircle2 } from "lucide-react";

// --- 인터페이스 정의 ---
interface TmsRecord {
  PH_VU: number;
  TOC_VU: number;
  TN_VU: number;
  TP_VU: number;
  SS_VU: number;
}

interface WeatherData {
  RN_15m: number;
}

interface AlertProps {
  latestValues: TmsRecord | null;
  latestWeather: WeatherData | null;
}

export default function Row3Alerts({ latestValues, latestWeather }: AlertProps) {
  // 1. pH 이상관측 로직 (6.0 이하 또는 8.0 이상)
  const phValue = latestValues?.PH_VU ?? 7.0;
  const isPhDanger = phValue <= 5.8 || phValue >= 8.5;

  // 2. TOC/T-N/T-P 로직
  const tocValue = latestValues?.TOC_VU ?? 0;
  const tnValue = latestValues?.TN_VU ?? 0;
  const tpValue = latestValues?.TP_VU ?? 0;
  const ssValue = latestValues ?.SS_VU ?? 0;

  const isTocDanger = tocValue >= 15;
  const isTnDanger = tnValue >= 20;
  const isTpDanger = tpValue >= 0.5;
  const isSSDanger = ssValue >= 10;
  const isTmsDanger = isTocDanger || isTnDanger || isTpDanger;

  // 3. 강우량 로직 (10mm 이상)
  const rainValue = latestWeather?.RN_15m ?? 0;
  const isRainDanger = rainValue >= 10;

  const alerts = [
    { 
      id: 1, 
      title: "pH 이상관측 (6.0~8.0)", 
      valueText: `현재: ${phValue.toFixed(2)}`,
      status: isPhDanger ? "danger" : "normal", 
      icon: <Activity className="w-4 h-4" /> 
    },
    { 
      id: 2, 
      title: "방류수질 기준 (TOC/TN/TP/SS)", 
      details: [
        { name: "TOC", danger: isTocDanger, val: tocValue, limit: 15 },
        { name: "T-N", danger: isTnDanger, val: tnValue, limit: 20 },
        { name: "T-P", danger: isTpDanger, val: tpValue, limit: 0.5 },
        { name: "SS", danger: isSSDanger, val: ssValue, limit: 10 }
      ],
      status: isTmsDanger ? "danger" : "normal", 
      icon: <AlertTriangle className="w-4 h-4" /> 
    },
    { 
      id: 3, 
      title: "강우 감지 (기준: 10mm)", 
      valueText: `현재: ${rainValue.toFixed(1)}mm`,
      status: isRainDanger ? "danger" : "normal", 
      icon: <DatabaseZap className="w-4 h-4" /> 
    },
  ];

  const handleActionClick = (title: string) => {
    const event = new CustomEvent("setMemoInput", { 
      detail: { text: `[이벤트 알림: ${title}] 관련 조치 사항을 입력하세요.` } 
    });
    window.dispatchEvent(event);
  };

  return (
    <div className="bg-slate-800/40 p-4 rounded-3xl border border-white/10 h-full flex flex-col">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest">Event Detection</h3>
        <div className="flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-red-500 animate-ping"></span>
          <span className="text-[10px] text-red-400 font-black">REAL-TIME</span>
        </div>
      </div>
      
      <div className="space-y-3 flex-1 overflow-y-auto pr-1 custom-scrollbar">
        {alerts.map((alert, index) => (
          <motion.div
            initial={{ opacity: 0, y: 5 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            key={alert.id}
            className={`flex items-center justify-between p-3 rounded-2xl border transition-all group/item ${
              alert.status === 'danger' 
                ? 'bg-red-500/10 border-red-500/30' 
                : 'bg-slate-900/40 border-white/5'
            }`}
          >
            <div className="flex flex-col gap-1.5">
              <div className="flex items-center gap-2.5">
                <div className={alert.status === 'danger' ? 'text-red-500' : 'text-emerald-500'}>
                  {alert.status === 'danger' ? alert.icon : <CheckCircle2 className="w-4 h-4" />}
                </div>
                <span className={`text-[13px] font-bold ${alert.status === 'danger' ? 'text-red-200' : 'text-slate-200'}`}>
                  {alert.title}
                </span>
              </div>
              
              <div className="flex items-center gap-2 ml-6">
                {alert.details ? (
                  <div className="flex gap-3">
                    {alert.details.map((d) => (
                      <div key={d.name} className="flex items-center gap-1.5">
                        <div className={`w-1.5 h-1.5 rounded-full ${d.danger ? 'bg-red-500 animate-pulse' : 'bg-emerald-500'}`} />
                        <span className={`text-[10px] font-mono ${d.danger ? 'text-red-400' : 'text-slate-500'}`}>
                          {d.name}
                        </span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <span className={`text-[10px] font-mono ${alert.status === 'danger' ? 'text-red-400' : 'text-slate-500'}`}>
                    {alert.valueText}
                  </span>
                )}
              </div>
            </div>
            
            <button 
              onClick={() => handleActionClick(alert.title)}
              className="px-3 py-1.5 rounded-xl bg-white/5 hover:bg-blue-600 text-slate-400 hover:text-white text-[10px] font-bold transition-all border border-white/10 opacity-0 group-hover/item:opacity-100"
            >
              조치하기
            </button>
          </motion.div>
        ))}
      </div>
    </div>
  );
}