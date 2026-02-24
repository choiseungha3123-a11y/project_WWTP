"use client";

import { useState, useEffect } from "react";
import useSWR from "swr";

// --- 인터페이스 정의 ---
interface TmsRecord { SYS_TIME: string; TOC_VU: number; PH_VU: number; SS_VU: number; FLUX_VU: number; TN_VU: number; TP_VU: number; }
interface WeatherRecord { SYS_TIME: string; TA: number; RN_15m: number; RN_60m?: number; RN_12H?: number; RN_DAY?: number; HM?: number; TD?: number; distance?: number; }
interface FlowRecord { SYS_TIME?: string; flowTime?: string; Q_in?: number; flowValue?: number; level_TankA?: number; level_TankB?: number; flow_TankA?: number; flow_TankB?: number; }
type BoardRecord = TmsRecord | WeatherRecord | FlowRecord;
interface BoardViewResponse { success: boolean; dataList: BoardRecord[][]; }
interface StatusCardProps { label: string; value: string | number; status: "normal" | "warning" | "danger"; time?: string | null; isError?: boolean; unit?: string; }

// ----------------------------------------------------------------------
// 2. 유틸리티 함수
// ----------------------------------------------------------------------

const formatTime = (timeStr?: string | null) => {
  if (!timeStr) return "-";
  if (timeStr.length === 14 && !timeStr.includes("-") && !isNaN(Number(timeStr))) {
    const y = timeStr.substring(0, 4);
    const m = timeStr.substring(4, 6);
    const d = timeStr.substring(6, 8);
    const h = timeStr.substring(8, 10);
    const min = timeStr.substring(10, 12);
    return `${y}-${m}-${d} ${h}:${min}`;
  }
  if (timeStr.includes("T")) return timeStr.replace("T", " ").substring(0, 16);
  return timeStr;
};

const roundVal = (val: number | undefined | null) => {
  if (val === undefined || val === null || isNaN(Number(val))) return "0.00";
  return Number(val).toFixed(2);
};

const formatNumberWithComma = (val: number | undefined | null) => {
  if (val === undefined || val === null || isNaN(Number(val))) return "0.00";
  return Number(val).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
};

const fetcher = async (url: string) => {
  const token = typeof window !== "undefined" ? localStorage.getItem("accessToken") : null;
  const response = await fetch(url, {
    method: "GET",
    headers: { "Content-Type": "application/json", ...(token ? { "Authorization": `Bearer ${token}` } : {}) },
  });
  if (response.status === 401) throw new Error("인증 세션이 만료되었습니다.");
  if (!response.ok) throw new Error("API 연결 실패");
  return response.json();
};

// ----------------------------------------------------------------------
// 3. 메인 컴포넌트
// ----------------------------------------------------------------------

export default function Row1Status() {
  const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL;
  const [isClient, setIsClient] = useState(false);

  useEffect(() => { setIsClient(true); }, []);

  // 주요 데이터 페칭 (30분 간격 - 필요에 따라 조정 가능)
  const { data: tmsRaw, error: tmsError } = useSWR<BoardViewResponse>(
    isClient ? `${API_BASE_URL}/api/board/boardView` : null, fetcher, { refreshInterval: 30 * 60 * 1000, revalidateOnFocus: true }
  );

  const dataList = tmsRaw?.success ? tmsRaw.dataList : [];

  // 데이터 추출 로직
  const tmsData = dataList.find((l): l is TmsRecord[] => l.length > 0 && 'TOC_VU' in l[0])?.slice(-1)[0] || null;
  const weatherData = dataList.find((l): l is WeatherRecord[] => l.length > 0 && 'TA' in l[0])?.slice(-1)[0] || null;
  const flowRawData = dataList.find((l): l is FlowRecord[] => l.length > 0 && ('Q_in' in l[0] || 'flowValue' in l[0]))?.slice(-1)[0] || null;

  const inflowValue = flowRawData ? (flowRawData.Q_in ?? flowRawData.flowValue ?? 0) : 0;
  const flowTime = flowRawData?.SYS_TIME || flowRawData?.flowTime;

  if (!isClient) return <div className="h-48 animate-pulse bg-slate-900/50 rounded-2xl" />;

  return (
    <div className="flex flex-col gap-4">
      {/* 이전의 "시스템 체크" 상단 바가 제거되었습니다. 
         이제 데이터 카드들이 바로 그리드로 표시됩니다.
      */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <StatusCard 
          label="유입유량"
          value={tmsError ? "Error" : (flowRawData ? formatNumberWithComma(inflowValue) : "Loading...")}
          status={tmsError ? "danger" : "normal"}
          time={flowTime}
          isError={!!tmsError}
          unit="㎥/h"
        />
        <StatusCard 
          label="기온 | 강우"
          value={weatherData ? `${roundVal(weatherData.TA)}°C | ${weatherData.RN_15m > 0 ? `${roundVal(weatherData.RN_15m)}mm` : "강우없음"}` : (tmsError ? "Error" : "Loading...")}
          status={(weatherData && weatherData.RN_15m > 5) ? "danger" : (weatherData && weatherData.RN_15m > 0) ? "warning" : (tmsError ? "danger" : "normal")}
          time={weatherData?.SYS_TIME}
          isError={!!tmsError}
        />
        <StatusCard 
          label="pH | FLUX"
          value={tmsData ? `pH ${roundVal(tmsData.PH_VU)} | ${formatNumberWithComma(tmsData.FLUX_VU)}` : (tmsError ? "Error" : "Loading...")}
          status={(tmsData && (tmsData.PH_VU > 8 || tmsData.PH_VU < 6)) ? "warning" : (tmsError ? "danger" : "normal")}
          time={tmsData?.SYS_TIME}
          isError={!!tmsError}
        />
        <StatusCard 
          label="TMS (TOC/TN/TP/SS)"
          value={tmsData ? `${roundVal(tmsData.TOC_VU)} / ${roundVal(tmsData.TN_VU)} / ${roundVal(tmsData.TP_VU)} / ${roundVal(tmsData.SS_VU)}` : (tmsError ? "Error" : "Loading...")}
          status={tmsError ? "danger" : "normal"}
          time={tmsData?.SYS_TIME}
          isError={!!tmsError}
        />
      </div>
    </div>
  );
}

function StatusCard({ label, value, status, time, isError = false, unit = "" }: StatusCardProps) {
  return (
    <div className="bg-slate-800/40 p-6 rounded-2xl border border-white/5 flex flex-col justify-between relative shadow-sm hover:border-white/10 transition-colors group h-32">
      <div className="flex justify-between items-start mb-2">
        <h3 className="text-slate-400 text-sm font-bold uppercase tracking-widest group-hover:text-slate-300 transition-colors">{label}</h3>
        <span className={`w-2.5 h-2.5 rounded-full ${status === 'warning' ? 'bg-orange-500 animate-pulse' : status === 'danger' ? 'bg-red-500 animate-ping' : 'bg-emerald-500'}`}></span>
      </div>
      <div className={`text-2xl font-black tracking-tight ${status === 'warning' ? 'text-orange-400' : status === 'danger' ? 'text-red-400' : 'text-emerald-400'}`}>
        {value} <span className="text-sm font-medium text-slate-500 ml-1">{unit}</span>
      </div>
      <div className="text-[11px] text-slate-600 font-mono tracking-tight mt-auto text-right uppercase pt-2 border-t border-white/5">
        {isError ? "Fetch Failed" : (time ? `${formatTime(time)} Update` : "No Data")}
      </div>
    </div>
  );
}