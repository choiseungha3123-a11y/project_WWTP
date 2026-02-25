"use client";

import { useMemo, useState, useEffect } from "react";
import useSWR from "swr";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from "recharts";

// ----------------------------------------------------------------------
// 1. 인터페이스 정의
// ----------------------------------------------------------------------

interface TmsRecord {
  SYS_TIME: string;
  TOC_VU: number;
  PH_VU: number;
  SS_VU: number;
  FLUX_VU: number;
  TN_VU: number;
  TP_VU: number;
}

interface FlowRecord {
  SYS_TIME: string;
  Q_in: number;
  flow_TankA?: number;
  flow_TankB?: number;
  level_TankA?: number;
  level_TankB?: number;
}

interface MergedData {
  displayTime: string;
  toc_A?: number; ph_A?: number; ss_A?: number;
  flux_A?: number; tn_A?: number; tp_A?: number;
  Q_in_A?: number;
  toc_P?: number; ph_P?: number; ss_P?: number;
  flux_P?: number; tn_P?: number; tp_P?: number;
  Q_in_P?: number;
}

interface LatestValues extends Partial<TmsRecord> {
  Q_in?: number;
}

interface BoardViewResponse {
  success: boolean;
  dataList: [TmsRecord[], TmsRecord[], FlowRecord[], FlowRecord[]]; 
}

// ----------------------------------------------------------------------
// 2. 유틸리티 함수
// ----------------------------------------------------------------------

const fetcher = async (url: string) => {
  const token = typeof window !== "undefined" ? localStorage.getItem("accessToken") : null;
  const res = await fetch(url, {
    headers: {
      Authorization: token ? `Bearer ${token}` : "",
      "Content-Type": "application/json",
    },
  });
  if (!res.ok) throw new Error("차트 데이터 로드 실패");
  return res.json();
};

const formatDisplayTime = (timeStr: string) => {
  if (!timeStr) return "";
  if (timeStr.includes("T")) return timeStr.split("T")[1].substring(0, 5);
  if (timeStr.length >= 12) return `${timeStr.substring(8, 10)}:${timeStr.substring(10, 12)}`;
  return timeStr;
};

// ----------------------------------------------------------------------
// 3. 메인 컴포넌트
// ----------------------------------------------------------------------

export default function Row1Charts() {
  const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL;
  const [isClient, setIsClient] = useState(false);

  useEffect(() => { setIsClient(true); }, []);

  const { data: rawData, error, isLoading } = useSWR<BoardViewResponse>(
    isClient ? `${API_BASE_URL}/api/board/boardView` : null,
    fetcher,
    { refreshInterval: 30 * 60 * 1000 }
  );

  const { chartData, latestValues } = useMemo(() => {
    if (!rawData?.success || !rawData.dataList || rawData.dataList.length < 4) {
      return { chartData: [] as MergedData[], latestValues: null as LatestValues | null };
    }

    const [actualTms, predictTms, actualFlow, predictFlow] = rawData.dataList;
    const mergedMap = new Map<string, MergedData>();

    actualTms.forEach((item) => {
      const displayTime = formatDisplayTime(item.SYS_TIME);
      mergedMap.set(displayTime, {
        displayTime,
        toc_A: item.TOC_VU, ph_A: item.PH_VU, ss_A: item.SS_VU,
        flux_A: item.FLUX_VU, tn_A: item.TN_VU, tp_A: item.TP_VU,
      });
    });

    actualFlow.forEach((item) => {
      const displayTime = formatDisplayTime(item.SYS_TIME);
      const existing = mergedMap.get(displayTime) || { displayTime };
      mergedMap.set(displayTime, { ...existing, Q_in_A: item.Q_in });
    });

    predictTms.forEach((item) => {
      const displayTime = formatDisplayTime(item.SYS_TIME);
      const existing = mergedMap.get(displayTime) || { displayTime };
      mergedMap.set(displayTime, {
        ...existing,
        toc_P: item.TOC_VU, ph_P: item.PH_VU, ss_P: item.SS_VU,
        flux_P: item.FLUX_VU, tn_P: item.TN_VU, tp_P: item.TP_VU,
      });
    });

    predictFlow.forEach((item) => {
      const displayTime = formatDisplayTime(item.SYS_TIME);
      const existing = mergedMap.get(displayTime) || { displayTime };
      mergedMap.set(displayTime, { ...existing, Q_in_P: item.Q_in });
    });

    const sortedData = Array.from(mergedMap.values()).sort((a, b) => 
      a.displayTime.localeCompare(b.displayTime)
    );
    
    const lastTms = actualTms.length > 0 ? actualTms[actualTms.length - 1] : null;
    const lastFlow = actualFlow.length > 0 ? actualFlow[actualFlow.length - 1] : null;

    const latest: LatestValues | null = lastTms ? {
      ...lastTms,
      Q_in: lastFlow?.Q_in
    } : null;

    return { chartData: sortedData, latestValues: latest };
  }, [rawData]);

  if (isLoading || !isClient) return <div className="p-4 text-slate-500">Loading...</div>;
  if (error) return <div className="p-4 text-red-400">Error!</div>;

  const renderRow = (
    title: string, 
    color: string, 
    keys: { actual: string; predict: string }, 
    latestVal: number | undefined, 
    unit: string = ""
  ) => (
    <div className="flex w-full items-stretch mb-5 last:mb-0 h-28">
      {/* 왼쪽 카드: 배경을 slate-700으로 밝게 조정하고 테두리 강조 추가 */}
      <div 
        className="w-32 bg-slate-700/80 rounded-l-2xl border-y border-l border-white/20 flex flex-col justify-center items-center p-3 shrink-0 shadow-lg"
        style={{ borderLeft: `4px solid ${color}` }} // 항목별 고유 색상으로 왼쪽 포인트 강조
      >
        <span className="text-xs text-slate-200 font-bold mb-1.5 uppercase tracking-wider">{title}</span>
        <span className="text-2xl font-black tracking-tighter drop-shadow-md" style={{ color }}>
          {latestVal !== undefined && latestVal !== null ? latestVal.toFixed(2) : "-"}
        </span>
        {unit && <span className="text-[10px] text-slate-300 mt-1 font-semibold">{unit}</span>}
      </div>

      {/* 오른쪽 그래프 영역 */}
      <div className="flex-1 ml-1 bg-slate-800/40 rounded-r-2xl border border-white/10 p-2 overflow-hidden shadow-sm">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 5, right: 10, left: -30, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#ffffff08" vertical={false} />
            <XAxis dataKey="displayTime" tick={{ fontSize: 9, fill: '#94a3b8' }} stroke="#475569" interval="preserveStartEnd" />
            <YAxis tick={{ fontSize: 9, fill: '#94a3b8' }} stroke="#475569" domain={['auto', 'auto']} />
            <Tooltip 
              contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', fontSize: '11px', borderRadius: '8px', color: '#f8fafc' }} 
              itemStyle={{ fontWeight: 'bold' }}
            />
            <Line type="monotone" dataKey={keys.actual} stroke={color} strokeWidth={3} dot={false} connectNulls isAnimationActive={false} />
            <Line type="monotone" dataKey={keys.predict} stroke={color} strokeWidth={2} strokeDasharray="4 4" dot={false} connectNulls isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );

  return (
    <div className="flex flex-col w-full h-full bg-slate-900/40 p-4 overflow-y-auto">
      {renderRow("유입유량", "#10b981", { actual: "Q_in_A", predict: "Q_in_P" }, latestValues?.Q_in, "m³/hr")}
      {renderRow("TOC", "#ef4444", { actual: "toc_A", predict: "toc_P" }, latestValues?.TOC_VU, "mg/L")}
      {renderRow("pH", "#3b82f6", { actual: "ph_A", predict: "ph_P" }, latestValues?.PH_VU)}
      {renderRow("SS", "#f59e0b", { actual: "ss_A", predict: "ss_P" }, latestValues?.SS_VU, "mg/L")}
      {renderRow("FLUX", "#ec4899", { actual: "flux_A", predict: "flux_P" }, latestValues?.FLUX_VU, "m³/hr")}
      {renderRow("T-N", "#8b5cf6", { actual: "tn_A", predict: "tn_P" }, latestValues?.TN_VU, "mg/L")}
      {renderRow("T-P", "#f97316", { actual: "tp_A", predict: "tp_P" }, latestValues?.TP_VU, "mg/L")}
    </div>
  );
}