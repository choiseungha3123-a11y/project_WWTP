"use client";

// 1. 사용하지 않는 motion, PencilLine 제거
import { AlertTriangle, Activity, DatabaseZap, CheckCircle2 } from "lucide-react";

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
  // 데이터 가독성을 위한 기본값 처리
  const phValue = latestValues?.PH_VU ?? 7.0;
  const tocValue = latestValues?.TOC_VU ?? 0;
  const tnValue = latestValues?.TN_VU ?? 0;
  const tpValue = latestValues?.TP_VU ?? 0;
  const ssValue = latestValues?.SS_VU ?? 0;
  const rainValue = latestWeather?.RN_15m ?? 0;

  // 위험 여부 판단 로직 (선언된 변수 적극 활용)
  const isPhDanger = phValue <= 5.8 || phValue >= 8.5;
  const isTocDanger = tocValue >= 15;
  const isTnDanger = tnValue >= 20;
  const isTpDanger = tpValue >= 0.5;
  const isSSDanger = ssValue >= 10;
  
  // 2. 선언만 되었던 변수들을 status 판별에 사용
  const isTmsDanger = isTocDanger || isTnDanger || isTpDanger || isSSDanger;
  const isRainDanger = rainValue >= 10;

  const alerts = [
    { 
      id: 1, 
      title: "pH 이상관측 (5.8 ~ 8.5)", 
      valueText: `현재: ${phValue.toFixed(2)}`,
      status: isPhDanger ? "danger" : "normal", 
      icon: <Activity className="w-4 h-4" /> 
    },
    { 
      id: 2, 
      title: "방류수질 기준 (TOC/TN/TP/SS)", 
      details: [
        { name: "TOC", danger: isTocDanger, val: tocValue },
        { name: "T-N", danger: isTnDanger, val: tnValue },
        { name: "T-P", danger: isTpDanger, val: tpValue },
        { name: "SS", danger: isSSDanger, val: ssValue }
      ],
      status: isTmsDanger ? "danger" : "normal", // 불필요한 중복 계산 제거
      icon: <AlertTriangle className="w-4 h-4" />
    },
    { 
      id: 3, 
      title: "강우 감지 (기준: 10mm)", 
      valueText: `현재 강수량: ${rainValue.toFixed(1)}mm`,
      status: isRainDanger ? "danger" : "normal", // 불필요한 중복 계산 제거
      icon: <DatabaseZap className="w-4 h-4" />
    },
  ];

  return (
    <div className="bg-slate-800/40 p-5 rounded-3xl border border-white/10 h-full flex flex-col shadow-inner">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest">Event Detection</h3>
        <div className="flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-red-500 animate-ping"></span>
          <span className="text-[10px] text-red-400 font-black">REAL-TIME</span>
        </div>
      </div>
      
      <div className="flex flex-col gap-3 flex-1 overflow-y-auto custom-scrollbar">
        {alerts.map((alert) => (
          <div 
            key={alert.id} 
            className={`p-3.5 rounded-2xl border transition-colors ${
              alert.status === 'danger' 
                ? 'bg-red-500/10 border-red-500/30 shadow-[0_0_15px_rgba(239,68,68,0.1)]' 
                : 'bg-slate-900/60 border-white/5'
            }`}
          >
            <div className="flex items-center gap-2.5 mb-2">
              <div className={alert.status === 'danger' ? 'text-red-500' : 'text-emerald-500'}>
                {alert.status === 'danger' ? alert.icon : <CheckCircle2 className="w-4 h-4" />}
              </div>
              <span className="text-[13px] font-bold text-slate-200">{alert.title}</span>
            </div>
            
            <div className="ml-7">
              {alert.details ? (
                <div className="grid grid-cols-2 gap-x-4 gap-y-1">
                  {alert.details.map((d) => (
                    <div key={d.name} className="flex items-center justify-between border-b border-white/5 pb-0.5">
                      <span className="text-[11px] text-slate-500 font-medium">{d.name}</span>
                      <span className={`text-[11px] font-mono font-bold ${d.danger ? 'text-red-400' : 'text-emerald-400'}`}>
                        {d.val.toFixed(2)}
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <span className={`text-[11px] font-medium ${alert.status === 'danger' ? 'text-red-300' : 'text-slate-400'}`}>
                  {alert.valueText}
                </span>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}