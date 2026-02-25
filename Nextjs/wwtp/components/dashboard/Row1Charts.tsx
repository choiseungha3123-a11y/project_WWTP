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

interface BoardViewResponse {
  success: boolean;
  dataList: TmsRecord[][];
}

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
export default function RowCharts() {
  const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL;
  const [isClient, setIsClient] = useState(false);

  useEffect(() => { setIsClient(true); }, []);

  const { data: rawData, error, isLoading } = useSWR<BoardViewResponse>(
    isClient ? `${API_BASE_URL}/api/board/boardView` : null,
    fetcher,
    { refreshInterval: 30 * 60 * 1000 }
  );

  // --- 데이터 병합 및 최신값 추출 ---
  const { chartData, latestValues } = useMemo(() => {
    if (!rawData?.success || !rawData.dataList || rawData.dataList.length < 2) {
      return { chartData: [], latestValues: null };
    }

    const actualList = rawData.dataList[0];
    const predictList = rawData.dataList[1];
    const mergedMap = new Map();

    // 실측 데이터 처리
    actualList.forEach((item) => {
      const displayTime = formatDisplayTime(item.SYS_TIME);
      mergedMap.set(displayTime, {
        displayTime,
        toc_A: item.TOC_VU, ph_A: item.PH_VU, ss_A: item.SS_VU,
        flux_A: item.FLUX_VU, tn_A: item.TN_VU, tp_A: item.TP_VU,
      });
    });

    // 예측 데이터 병합
    predictList.forEach((item) => {
      const displayTime = formatDisplayTime(item.SYS_TIME);
      const existing = mergedMap.get(displayTime) || { displayTime };
      mergedMap.set(displayTime, {
        ...existing,
        toc_P: item.TOC_VU, ph_P: item.PH_VU, ss_P: item.SS_VU,
        flux_P: item.FLUX_VU, tn_P: item.TN_VU, tp_P: item.TP_VU,
      });
    });

    const sortedData = Array.from(mergedMap.values()).sort((a, b) => a.displayTime.localeCompare(b.displayTime));
    
    // 실측 데이터의 마지막 객체를 최신값으로 사용
    const latest = actualList.length > 0 ? actualList[actualList.length - 1] : null;

    return { chartData: sortedData, latestValues: latest };
  }, [rawData]);

  if (isLoading || !isClient) return <div className="p-4 text-slate-500">Loading...</div>;
  if (error) return <div className="p-4 text-red-400">Error!</div>;

  // 개별 항목 렌더링 함수
  const renderRow = (title: string, color: string, keys: { actual: string; predict: string }, latestVal: number | undefined, unit: string = "") => (
    <div className="flex w-full items-stretch mb-4 last:mb-0 h-25">
      {/* 왼쪽: 최신 데이터 카드 */}
      <div className="w-25 bg-slate-800/60 rounded-l-xl border-y border-l border-white/10 flex flex-col justify-center items-center p-2 shrink-0">
        <span className="text-[10px] text-slate-400 font-bold mb-1">{title}</span>
        <span className="text-[16px] font-black" style={{ color }}>
          {latestVal !== undefined ? latestVal.toFixed(2) : "-"}
        </span>
        {unit && <span className="text-[9px] text-slate-500">{unit}</span>}
      </div>

      {/* 오른쪽: 그래프 (5px 간격) */}
      <div className="flex-1 ml-1.25 bg-slate-800/30 rounded-r-xl border border-white/5 p-2 overflow-hidden">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 5, right: 5, left: -35, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#ffffff05" vertical={false} />
            <XAxis dataKey="displayTime" tick={{ fontSize: 8 }} stroke="#475569" interval="preserveStartEnd" />
            <YAxis tick={{ fontSize: 8 }} stroke="#475569" domain={['auto', 'auto']} />
            <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: 'none', fontSize: '10px' }} />
            <Line type="monotone" dataKey={keys.actual} stroke={color} strokeWidth={2} dot={false} connectNulls isAnimationActive={false} />
            <Line type="monotone" dataKey={keys.predict} stroke={color} strokeWidth={1.5} strokeDasharray="3 3" dot={false} connectNulls isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );

  return (
    <div className="flex flex-col w-full h-full bg-slate-900/20 p-2 overflow-y-auto">
      {renderRow("TOC", "#ef4444", { actual: "toc_A", predict: "toc_P" }, latestValues?.TOC_VU, "mg/L")}
      {renderRow("pH", "#3b82f6", { actual: "ph_A", predict: "ph_P" }, latestValues?.PH_VU)}
      {renderRow("SS", "#f59e0b", { actual: "ss_A", predict: "ss_P" }, latestValues?.SS_VU, "mg/L")}
      {renderRow("유량", "#ec4899", { actual: "flux_A", predict: "flux_P" }, latestValues?.FLUX_VU, "m³/hr")}
      {renderRow("T-N", "#8b5cf6", { actual: "tn_A", predict: "tn_P" }, latestValues?.TN_VU, "mg/L")}
      {renderRow("T-P", "#f97316", { actual: "tp_A", predict: "tp_P" }, latestValues?.TP_VU, "mg/L")}
    </div>
  );
}