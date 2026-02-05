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
  const [tmsData, setTmsData] = useState<TmsRecord | null>(null);
  const [weatherData, setWeatherData] = useState<WeatherRecord | null>(null);
  const [isSystemOk, setIsSystemOk] = useState<boolean>(false);
  const [isClient, setIsClient] = useState(false);

  // 1. TMS 및 날씨 데이터를 가져오는 함수 (30분 주기)
  const fetchTmsAndWeather = async () => {
    try {
      const response = await fetch("http://10.125.121.176:8081/api/tmsOrigin/tmsList");
      const json = await response.json();

      if (json.success && json.dataList) {
        // TMS: dataList[0]의 마지막 값
        const tmsList = json.dataList[0] as TmsRecord[];
        if (tmsList?.length > 0) setTmsData(tmsList[tmsList.length - 1]);

        // 날씨: dataList[1]의 마지막 값
        const weatherList = json.dataList[1] as WeatherRecord[];
        if (weatherList?.length > 0) setWeatherData(weatherList[weatherList.length - 1]);
      }
    } catch (error) {
      console.error("TMS Data fetch error:", error);
    }
  };

  // 2. 시스템 상태를 체크하는 함수 (30초 주기)
  const fetchHealthCheck = async () => {
    try {
      const response = await fetch("http://10.125.121.176:8081/api/member/health");
      const json = await response.json();
      setIsSystemOk(json.success === true);
    } catch (error) {
      console.error("Health check error:", error);
      setIsSystemOk(false);
    }
  };

  useEffect(() => {
    setIsClient(true);

    // 컴포넌트 마운트 시 최초 1회 실행
    fetchTmsAndWeather();
    fetchHealthCheck();

    // 각각의 인터벌 설정
    const tmsTimer = setInterval(fetchTmsAndWeather, 30 * 60 * 1000); // 30분
    const healthTimer = setInterval(fetchHealthCheck, 30 * 1000);    // 30초

    // 클린업: 컴포넌트 언마운트 시 타이머 제거
    return () => {
      clearInterval(tmsTimer);
      clearInterval(healthTimer);
    };
  }, []);

  if (!isClient) return <div className="grid grid-cols-3 gap-4 h-24 animate-pulse bg-slate-900/50 rounded-2xl" />;

  const items = [
    { 
      label: "유입유량", 
      value: tmsData ? `${tmsData.FLUX_VU.toLocaleString()}` : "-", 
      status: "normal" 
    },
    { 
      label: "pH | FLUX", 
      value: tmsData ? `${tmsData.PH_VU.toFixed(1)} | ${tmsData.FLUX_VU}` : "-", 
      status: (tmsData && (tmsData.PH_VU > 8 || tmsData.PH_VU < 6)) ? "warning" : "normal" 
    },
    { 
      label: "TMS (TOC/TN/TP/SS)", 
      value: tmsData ? `${tmsData.TOC_VU.toFixed(1)} / ${tmsData.TN_VU.toFixed(1)} / ${tmsData.TP_VU.toFixed(1)} / ${tmsData.SS_VU.toFixed(1)}` : "-", 
      status: "normal" 
    },
    { 
      label: "기온 | 강우", 
      value: weatherData 
        ? `${weatherData.TA}°C | ${weatherData.RN_15m > 0 ? `${weatherData.RN_15m}mm` : "맑음"}` 
        : "-", 
      status: (weatherData && weatherData.RN_15m > 5) ? "danger" : (weatherData && weatherData.RN_15m > 0) ? "warning" : "normal" 
    },
    { 
      label: "데이터 상태", 
      value: tmsData ? "수신중" : "연결중", 
      status: tmsData ? "normal" : "warning" 
    },
    { 
      label: "시스템 체크", 
      value: isSystemOk ? "정상" : "점검필요", 
      status: isSystemOk ? "normal" : "danger" 
    },
  ];

  return (
    <div className="grid grid-cols-3 gap-4">
      {items.map((item, i) => (
        <div key={i} className="bg-slate-800/40 p-4 rounded-2xl border border-white/5 flex flex-col items-center justify-center">
          <span className="text-slate-400 text-[10px] mb-1 uppercase tracking-widest">{item.label}</span>
          <span className={`text-lg font-black ${
            item.status === 'warning' ? 'text-orange-400' : 
            item.status === 'danger' ? 'text-red-400' : 'text-emerald-400'
          }`}>
            {item.value}
          </span>
        </div>
      ))}
    </div>
  );
}