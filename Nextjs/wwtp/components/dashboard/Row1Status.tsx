"use client";

import { useState, useEffect } from "react";

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
  TA: number;      // 기온
  RN_15m: number;  // 강우량
}

export default function Row1Status() {
  const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL;

  const [tmsData, setTmsData] = useState<TmsRecord | null>(null);
  const [weatherData, setWeatherData] = useState<WeatherRecord | null>(null);
  const [isSystemOk, setIsSystemOk] = useState<boolean>(false);
  const [isClient, setIsClient] = useState(false);

  const fetchTmsAndWeather = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/tmsOrigin/tmsList`);
      const json = await response.json();

      if (json.success && json.dataList) {
        const tmsList = json.dataList[0] as TmsRecord[];
        if (tmsList?.length > 0) setTmsData(tmsList[tmsList.length - 1]);

        const weatherList = json.dataList[1] as WeatherRecord[];
        if (weatherList?.length > 0) setWeatherData(weatherList[weatherList.length - 1]);
      }
    } catch (error) {
      console.error("TMS Data fetch error:", error);
    }
  };

  const fetchHealthCheck = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/member/health`);
      const json = await response.json();
      setIsSystemOk(json.success === true);
    } catch (error) {
      console.error("Health check error:", error);
      setIsSystemOk(false);
    }
  };

  useEffect(() => {
    setIsClient(true);
    fetchTmsAndWeather();
    fetchHealthCheck();

    const tmsTimer = setInterval(fetchTmsAndWeather, 30 * 60 * 1000);
    const healthTimer = setInterval(fetchHealthCheck, 10 * 1000);

    return () => {
      clearInterval(tmsTimer);
      clearInterval(healthTimer);
    };
  }, []);

  if (!isClient) return <div className="grid grid-cols-3 gap-4 h-24 animate-pulse bg-slate-900/50 rounded-2xl" />;

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
    if (timeStr.includes("T")) {
      return timeStr.replace("T", " ").substring(0, 16);
    }
    return timeStr;
  };

  const items = [
    { 
      label: "유입유량", 
      value: tmsData ? `${tmsData.FLUX_VU.toLocaleString()}` : "Loading...", 
      status: "normal",
      time: tmsData?.SYS_TIME 
    },
    { 
      label: "pH | FLUX", 
      value: tmsData ? `${tmsData.PH_VU.toFixed(1)} | ${tmsData.FLUX_VU}` : "Loading...", 
      status: (tmsData && (tmsData.PH_VU > 8 || tmsData.PH_VU < 6)) ? "warning" : "normal",
      time: tmsData?.SYS_TIME
    },
    { 
      label: "TMS (TOC/TN/TP/SS)", 
      value: tmsData ? `${tmsData.TOC_VU.toFixed(1)} / ${tmsData.TN_VU.toFixed(1)} / ${tmsData.TP_VU.toFixed(1)} / ${tmsData.SS_VU.toFixed(1)}` : "Loading...", 
      status: "normal",
      time: tmsData?.SYS_TIME
    },
    { 
      label: "기온 | 강우", 
      value: weatherData 
        ? `${weatherData.TA}°C | ${weatherData.RN_15m > 0 ? `${weatherData.RN_15m}mm` : "맑음"}` 
        : "Loading...", 
      status: (weatherData && weatherData.RN_15m > 5) ? "danger" : (weatherData && weatherData.RN_15m > 0) ? "warning" : "normal",
      time: weatherData?.SYS_TIME 
    },
    { 
      label: "데이터 상태", 
      value: tmsData ? "수신중" : "연결중", 
      status: tmsData ? "normal" : "warning",
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
          
          {/* 상단 라벨 영역: 기존 text-[10px] -> text-sm font-bold 로 변경 */}
          <div className="flex justify-between items-start mb-1">
            <h3 className="text-slate-400 text-sm font-bold uppercase tracking-widest group-hover:text-slate-300 transition-colors">
              {item.label}
            </h3>
            
            {/* 상태 표시 점 (우측 상단) */}
            <span className={`w-2 h-2 rounded-full ${
                item.status === 'warning' ? 'bg-orange-500 animate-pulse' : 
                item.status === 'danger' ? 'bg-red-500 animate-ping' : 'bg-emerald-500'
            }`}></span>
          </div>

          {/* 데이터 값 영역: 글자가 커진 만큼 Value도 text-xl로 키움 */}
          <div className={`text-xl font-black mt-1 ${
            item.status === 'warning' ? 'text-orange-400' : 
            item.status === 'danger' ? 'text-red-400' : 'text-emerald-400'
          }`}>
            {item.value}
          </div>

          {/* 하단 시간 표시 영역 */}
          {item.time && (
            <div className="text-[10px] text-slate-600 font-mono tracking-tighter mt-auto text-right">
              {formatTime(item.time)} UPDATE
            </div>
          )}
        </div>
      ))}
    </div>
  );
}