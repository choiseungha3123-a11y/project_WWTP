"use client";

import { useState, useEffect } from "react";
import useSWR from "swr";

interface TmsRecord {
  SYS_TIME: string;
  TOC_VU: number;
  PH_VU: number;
  SS_VU: number;
  FLUX_VU: number;
  TN_VU: number;
  TP_VU: number;
}

interface WeatherRecord {
  SYS_TIME: string;
  TA: number;
  RN_15m: number;
}

const fetcher = async (url: string) => {
  const response = await fetch(url);
  if (!response.ok) throw new Error("API 연결 실패");
  return response.json();
};

export default function Row1Status() {
  const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL;
  const [isClient, setIsClient] = useState(false);

  const { data: tmsRaw, error: tmsError } = useSWR(
    `${API_BASE_URL}/api/tmsOrigin/tmsList`,
    fetcher,
    { 
      refreshInterval: 30 * 60 * 1000, 
      revalidateOnFocus: true 
    }
  );

  const { data: healthRaw } = useSWR(
    `${API_BASE_URL}/api/member/health`,
    fetcher,
    { refreshInterval: 10 * 1000 }
  );

  useEffect(() => {
    setIsClient(true);
  }, []);

  if (!isClient) return <div className="grid grid-cols-3 gap-4 h-24 animate-pulse bg-slate-900/50 rounded-2xl" />;

  const tmsData: TmsRecord | null = (tmsRaw?.success && tmsRaw.dataList?.[0]) 
    ? tmsRaw.dataList[0][tmsRaw.dataList[0].length - 1] 
    : null;

  const weatherData: WeatherRecord | null = (tmsRaw?.success && tmsRaw.dataList?.[1]) 
    ? tmsRaw.dataList[1][tmsRaw.dataList[1].length - 1] 
    : null;

  const isSystemOk = healthRaw?.success === true;

  const formatTime = (timeStr?: string) => {
    if (!timeStr) return "";
    if (timeStr.length === 14 && !timeStr.includes("-") && !isNaN(Number(timeStr))) {
      const y = timeStr.substring(0, 4);
      const m = timeStr.substring(4, 6);
      const d = timeStr.substring(6, 8);
      const h = timeStr.substring(8, 10);
      const min = timeStr.substring(10, 12);
      return `${y}-${m}-${d} ${h}:${min}`;
    }
    return timeStr.includes("T") ? timeStr.replace("T", " ").substring(0, 16) : timeStr;
  };

  const roundVal = (val: number) => val.toFixed(1);

  const items = [
    { 
      label: "유입유량", 
      value: tmsError ? "Error" : (tmsData ? `${tmsData.FLUX_VU.toLocaleString()}` : "Loading..."), 
      status: tmsError ? "danger" : "normal",
      time: tmsData?.SYS_TIME 
    },
    { 
      label: "pH | FLUX", 
      value: tmsData 
        ? `${roundVal(tmsData.PH_VU)} | ${tmsData.FLUX_VU.toLocaleString()}` 
        : (tmsError ? "Error" : "Loading..."), 
      status: (tmsData && (tmsData.PH_VU > 8 || tmsData.PH_VU < 6)) ? "warning" : (tmsError ? "danger" : "normal"),
      time: tmsData?.SYS_TIME
    },
    { 
      label: "TMS (TOC/TN/TP/SS)", 
      value: tmsData 
        ? `${roundVal(tmsData.TOC_VU)} / ${roundVal(tmsData.TN_VU)} / ${roundVal(tmsData.TP_VU)} / ${roundVal(tmsData.SS_VU)}` 
        : (tmsError ? "Error" : "Loading..."), 
      status: tmsError ? "danger" : "normal",
      time: tmsData?.SYS_TIME
    },
    { 
      label: "기온 | 강우", 
      value: weatherData 
        ? `${roundVal(weatherData.TA)}°C | ${weatherData.RN_15m > 0 ? `${roundVal(weatherData.RN_15m)}mm` : "맑음"}` 
        : (tmsError ? "Error" : "Loading..."), 
      status: (weatherData && weatherData.RN_15m > 5) ? "danger" : (weatherData && weatherData.RN_15m > 0) ? "warning" : (tmsError ? "danger" : "normal"),
      time: weatherData?.SYS_TIME 
    },
    { 
      label: "데이터 상태", 
      value: tmsError ? "수신실패" : (tmsData ? "수신중" : "연결중"), 
      status: tmsError ? "danger" : (tmsData ? "normal" : "warning"),
      time: tmsData?.SYS_TIME
    },
    { 
      label: "시스템 체크", 
      value: isSystemOk ? "정상" : "점검필요", 
      status: isSystemOk ? "normal" : "danger",
      time: null 
    },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {items.map((item, i) => (
        <div key={i} className="bg-slate-800/40 p-5 rounded-2xl border border-white/5 flex flex-col justify-between relative shadow-sm hover:border-white/10 transition-colors group h-28">
          <div className="flex justify-between items-start mb-1">
            <h3 className="text-slate-400 text-sm font-bold uppercase tracking-widest group-hover:text-slate-300 transition-colors">
              {item.label}
            </h3>
            <span className={`w-2 h-2 rounded-full ${
                item.status === 'warning' ? 'bg-orange-500 animate-pulse' : 
                item.status === 'danger' ? 'bg-red-500 animate-ping' : 'bg-emerald-500'
            }`}></span>
          </div>

          <div className={`text-xl font-black mt-1 ${
            item.status === 'warning' ? 'text-orange-400' : 
            item.status === 'danger' ? 'text-red-400' : 'text-emerald-400'
          }`}>
            {item.value}
          </div>

          {item.time && (
            <div className="text-[10px] text-slate-600 font-mono tracking-tighter mt-auto text-right uppercase">
              {tmsError ? "Fetch Failed" : `${formatTime(item.time)} Update`}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}