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
  Legend,
  ReferenceLine,
} from "recharts";

// ----------------------------------------------------------------------
// 1. 인터페이스 및 타입 정의
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
  SYS_TIME?: string;
  flowTime?: string;
  Q_in?: number;
  flowValue?: number;
}

type BoardRecord = TmsRecord | FlowRecord;

interface BoardViewResponse {
  success: boolean;
  dataList: BoardRecord[][];
}

// ----------------------------------------------------------------------
// 2. Fetcher 및 유틸리티
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

/**
 * 문자열 시간을 Date 객체로 변환 (API의 두 가지 형식 대응)
 */
const parseToDate = (timeStr: string) => {
  if (timeStr.includes("T")) return new Date(timeStr);
  if (timeStr.length === 14) {
    const formatted = timeStr.replace(
      /(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})/,
      "$1-$2-$3T$4:$5:$6"
    );
    return new Date(formatted);
  }
  return new Date();
};

// ----------------------------------------------------------------------
// 3. 메인 컴포넌트
// ----------------------------------------------------------------------

export default function Row2Charts() {
  const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL;
  const [isClient, setIsClient] = useState(false);
  
  // 현재 시점 (분기점)
  const currentTime = useMemo(() => new Date(), []);
  console.log('currentTime', currentTime);

  useEffect(() => {
    setIsClient(true);
  }, []);

  const { data: rawData, error, isLoading } = useSWR<BoardViewResponse>(
    isClient ? `${API_BASE_URL}/api/board/boardView` : null,
    fetcher,
    {
      refreshInterval: 30 * 60 * 1000,
      revalidateOnFocus: true,
    }
  );

  // --- 데이터 분류 ---

  const tmsItems = useMemo<TmsRecord[]>(() => {
    if (!rawData?.success || !rawData.dataList) return [];
    const found = rawData.dataList.find((list): list is TmsRecord[] => 
      list.length > 0 && 'TOC_VU' in list[0]
    );
    return found || [];
  }, [rawData]);

  const flowItems = useMemo<FlowRecord[]>(() => {
    if (!rawData?.success || !rawData.dataList) return [];
    const found = rawData.dataList.find((list): list is FlowRecord[] => 
      list.length > 0 && ('Q_in' in list[0] || 'flowValue' in list[0])
    );
    return found || [];
  }, [rawData]);

  // --- 차트 데이터 가공 (실측/예측 분리) ---

  const inflowChartData = useMemo(() => {
    return flowItems.map((d) => {
      const timeStr = d.SYS_TIME || d.flowTime || "";
      const itemDate = parseToDate(timeStr);
      const isPrediction = itemDate > currentTime;
      
      const displayTime = timeStr.includes("T") 
        ? timeStr.split("T")[1].substring(0, 5) 
        : timeStr.substring(8, 12).replace(/(\d{2})(\d{2})/, "$1:$2");
      
      const val = d.Q_in ?? d.flowValue ?? 0;

      return {
        displayTime,
        // 실측 데이터 (예측 시점 이후면 null)
        actual: !isPrediction ? val : null,
        // 예측 데이터 (실측의 마지막 데이터와 겹치게 처리하여 선 연결)
        predict: isPrediction || itemDate.getTime() === currentTime.setSeconds(0,0) ? val : null,
      };
    });
  }, [flowItems, currentTime]);

  const waterChartData = useMemo(() => {
    return tmsItems.map((d) => {
      const itemDate = parseToDate(d.SYS_TIME);
      const isPrediction = itemDate > currentTime;

      const displayTime = d.SYS_TIME.includes("T") 
        ? d.SYS_TIME.split("T")[1].substring(0, 5) 
        : d.SYS_TIME.substring(8, 12).replace(/(\d{2})(\d{2})/, "$1:$2");
        
      return {
        displayTime,
        toc_A: !isPrediction ? d.TOC_VU : null,
        toc_P: isPrediction || itemDate.getTime() === currentTime.setSeconds(0,0) ? d.TOC_VU : null,
        tn_A: !isPrediction ? d.TN_VU : null,
        tn_P: isPrediction || itemDate.getTime() === currentTime.setSeconds(0,0) ? d.TN_VU : null,
        tp_A: !isPrediction ? d.TP_VU : null,
        tp_P: isPrediction || itemDate.getTime() === currentTime.setSeconds(0,0) ? d.TP_VU : null,
        ss_A: !isPrediction ? d.SS_VU : null,
        ss_P: isPrediction || itemDate.getTime() === currentTime.setSeconds(0,0) ? d.SS_VU : null,
      };
    });
  }, [tmsItems, currentTime]);

  const dataDate = useMemo(() => {
    const firstTime = tmsItems[0]?.SYS_TIME || flowItems[0]?.SYS_TIME || flowItems[0]?.flowTime || "";
    if (firstTime.includes("T")) return firstTime.split("T")[0];
    if (firstTime.length >= 8) return `${firstTime.substring(0, 4)}-${firstTime.substring(4, 6)}-${firstTime.substring(6, 8)}`;
    return "";
  }, [tmsItems, flowItems]);

  if (isLoading) return <div className="h-full flex items-center justify-center text-slate-500 animate-pulse">데이터 분석 중...</div>;
  if (error) return <div className="h-full flex items-center justify-center text-red-400 text-xs">데이터 로드 실패</div>;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 w-full h-full min-h-0">
      
      {/* 1. 유입유량 차트 */}
      <div className="bg-slate-800/40 p-4 rounded-2xl border border-white/5 flex flex-col min-h-0 flex-1">
        <div className="flex justify-between items-center mb-2">
          <h3 className="text-[13px] font-bold text-blue-400">
            유입유량 변화 (실측/예측)
            <span className="ml-2 text-[10px] text-slate-500 font-normal">{dataDate}</span>
          </h3>
        </div>
        <div className="flex-1 w-full min-h-0"> 
          <ResponsiveContainer width="100%" height={362}>
            <LineChart data={inflowChartData} margin={{ top: 5, right: 10, left: -25, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#ffffff05" vertical={false} />
              <XAxis dataKey="displayTime" tick={{fontSize: 9}} stroke="#475569" />
              <YAxis tick={{fontSize: 9}} stroke="#475569" />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px', fontSize: '11px' }}
                itemStyle={{ color: '#fff', padding: '2px 0' }}
              />
              <Legend verticalAlign="top" align="right" wrapperStyle={{ fontSize: '9px', top: -10 }} />
              
              {/* 실측: 실선 */}
              <Line type="monotone" dataKey="actual" name="유입유량(실측)" stroke="#3b82f6" strokeWidth={2} dot={false} isAnimationActive={false} connectNulls />
              {/* 예측: 점선 */}
              <Line type="monotone" dataKey="predict" name="유입유량(예측)" stroke="#3b82f6" strokeWidth={2} strokeDasharray="5 5" dot={false} isAnimationActive={false} connectNulls />
              
              <ReferenceLine x={currentTime.toTimeString().substring(0, 5)} stroke="#ef4444" strokeDasharray="3 3" label={{ position: 'top', value: '현재', fill: '#ef4444', fontSize: 9 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* 2. 수질 통합 분석 차트 */}
      <div className="bg-slate-800/40 p-4 rounded-2xl border border-white/5 flex flex-col min-h-0 flex-1">
        <div className="flex justify-between items-center mb-2">
          <h3 className="text-[13px] font-bold text-emerald-400">
            수질 통합 분석 (실측/예측)
            <span className="ml-2 text-[10px] text-slate-500 font-normal">{dataDate}</span>
          </h3>
        </div>
        <div className="flex-1 w-full min-h-0">
          <ResponsiveContainer width="100%" height={362}>
            <LineChart data={waterChartData} margin={{ top: 5, right: 10, left: -25, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#ffffff05" vertical={false} />
              <XAxis dataKey="displayTime" tick={{fontSize: 9}} stroke="#475569" />
              <YAxis tick={{fontSize: 9}} stroke="#475569" />
              <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px', fontSize: '11px' }} />
              <Legend verticalAlign="top" align="right" wrapperStyle={{ fontSize: '9px', top: -10 }} />
              
              {/* TOC */}
              <Line type="monotone" dataKey="toc_A" name="TOC(실측)" stroke="#10b981" strokeWidth={1.5} dot={false} connectNulls isAnimationActive={false} />
              <Line type="monotone" dataKey="toc_P" name="TOC(예측)" stroke="#10b981" strokeWidth={1.5} strokeDasharray="4 4" dot={false} connectNulls isAnimationActive={false} />
              
              {/* T-N */}
              <Line type="monotone" dataKey="tn_A" name="T-N(실측)" stroke="#8b5cf6" strokeWidth={1.5} dot={false} connectNulls isAnimationActive={false} />
              <Line type="monotone" dataKey="tn_P" name="T-N(예측)" stroke="#8b5cf6" strokeWidth={1.5} strokeDasharray="4 4" dot={false} connectNulls isAnimationActive={false} />

              {/* T-P */}
              <Line type="monotone" dataKey="tp_A" name="T-P(실측)" stroke="#f59e0b" strokeWidth={1.5} dot={false} connectNulls isAnimationActive={false} />
              <Line type="monotone" dataKey="tp_P" name="T-P(예측)" stroke="#f59e0b" strokeWidth={1.5} strokeDasharray="4 4" dot={false} connectNulls isAnimationActive={false} />

              <ReferenceLine x={currentTime.toTimeString().substring(0, 5)} stroke="#ef4444" strokeDasharray="3 3" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

    </div>
  );
}