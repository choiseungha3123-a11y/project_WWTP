"use client";

import { useMemo, useState, useEffect } from "react";
import useSWR from "swr";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, Legend } from "recharts";

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

// dataList 내부의 각 배열 요소가 가질 수 있는 타입을 정의합니다.
type BoardRecord = TmsRecord | FlowRecord;

interface BoardViewResponse {
  success: boolean;
  dataList: BoardRecord[][];
}

// ----------------------------------------------------------------------
// 2. Fetcher 함수
// ----------------------------------------------------------------------

const fetcher = async (url: string) => {
  const token = typeof window !== "undefined" ? localStorage.getItem("accessToken") : null;
  const res = await fetch(url, {
    headers: {
      "Authorization": token ? `Bearer ${token}` : "",
      "Content-Type": "application/json",
    },
  });
  if (!res.ok) throw new Error("차트 데이터 로드 실패");
  return res.json();
};

// ----------------------------------------------------------------------
// 3. 메인 컴포넌트
// ----------------------------------------------------------------------

export default function Row3Charts() {
  const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL;
  const [isClient, setIsClient] = useState(false);

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

  // --- 데이터 분류 로직 (Type Guards 적용) ---

  // 1. 수질 데이터 추출 (TOC_VU 키를 포함하는 배열 탐색)
  const tmsItems = useMemo<TmsRecord[]>(() => {
    if (!rawData?.success || !rawData.dataList) return [];
    const found = rawData.dataList.find((list): list is TmsRecord[] => 
      list.length > 0 && 'TOC_VU' in list[0]
    );
    return found || [];
  }, [rawData]);

  // 2. 유입유량 데이터 추출 (Q_in 또는 flowValue 키를 포함하는 배열 탐색)
  const flowItems = useMemo<FlowRecord[]>(() => {
    if (!rawData?.success || !rawData.dataList) return [];
    const found = rawData.dataList.find((list): list is FlowRecord[] => 
      list.length > 0 && ('Q_in' in list[0] || 'flowValue' in list[0])
    );
    return found || [];
  }, [rawData]);

  // --- 차트 데이터 가공 ---

  // 유입유량 차트용 (Q_in / flowValue 통합 처리)
  const inflowChartData = useMemo(() => {
    return flowItems.map((d) => {
      const time = d.SYS_TIME || d.flowTime || "";
      // 시간 포맷팅: HH:mm 추출
      const displayTime = time.includes("T") 
        ? time.split("T")[1].substring(0, 5) 
        : time.substring(8, 12).replace(/(\d{2})(\d{2})/, "$1:$2");
      
      return {
        displayTime,
        value: d.Q_in ?? d.flowValue ?? 0,
      };
    });
  }, [flowItems]);

  // 수질 차트용
  const waterChartData = useMemo(() => {
    return tmsItems.map((d) => {
      const displayTime = d.SYS_TIME.includes("T") 
        ? d.SYS_TIME.split("T")[1].substring(0, 5) 
        : d.SYS_TIME.substring(8, 12).replace(/(\d{2})(\d{2})/, "$1:$2");
        
      return {
        ...d,
        displayTime,
      };
    });
  }, [tmsItems]);

  // 차트 상단 날짜 표시용
  const dataDate = useMemo(() => {
    const firstTime = tmsItems[0]?.SYS_TIME || flowItems[0]?.SYS_TIME || flowItems[0]?.flowTime || "";
    if (firstTime.includes("T")) return firstTime.split("T")[0];
    if (firstTime.length >= 8) return `${firstTime.substring(0, 4)}-${firstTime.substring(4, 6)}-${firstTime.substring(6, 8)}`;
    return "";
  }, [tmsItems, flowItems]);

  if (isLoading) return <div className="h-full flex items-center justify-center text-slate-500 animate-pulse">차트 데이터 동기화 중...</div>;
  if (error) return <div className="h-full flex items-center justify-center text-red-400 text-xs">차트 데이터를 불러올 수 없습니다.</div>;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 w-full h-full min-h-0">
      
      {/* 1. 유입유량(Inflow) 차트 */}
      <div className="bg-slate-800/40 p-4 rounded-2xl border border-white/5 flex flex-col min-h-0 flex-1">
        <div className="flex justify-between items-center mb-2">
          <h3 className="text-[13px] font-bold text-blue-400">
            유입유량 변화
            <span className="ml-2 text-[10px] text-slate-500 font-normal">{dataDate}</span>
          </h3>
        </div>
        <div className="flex-1 w-full min-h-0"> 
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={inflowChartData} margin={{ top: 5, right: 10, left: -25, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#ffffff05" vertical={false} />
              <XAxis dataKey="displayTime" tick={{fontSize: 9}} stroke="#475569" />
              <YAxis tick={{fontSize: 9}} stroke="#475569" />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px', fontSize: '11px' }}
                itemStyle={{ color: '#fff', padding: '2px 0' }}
              />
              <Legend verticalAlign="top" align="right" wrapperStyle={{ fontSize: '9px', top: -10 }} />
              <Line type="monotone" dataKey="value" name="유입유량" stroke="#3b82f6" strokeWidth={2} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* 2. 수질 통합 분석 차트 */}
      <div className="bg-slate-800/40 p-4 rounded-2xl border border-white/5 flex flex-col min-h-0 flex-1">
        <div className="flex justify-between items-center mb-2">
          <h3 className="text-[13px] font-bold text-emerald-400">
            수질 통합 분석
            <span className="ml-2 text-[10px] text-slate-500 font-normal">{dataDate}</span>
          </h3>
        </div>
        <div className="flex-1 w-full min-h-0">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={waterChartData} margin={{ top: 5, right: 10, left: -25, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#ffffff05" vertical={false} />
              <XAxis dataKey="displayTime" tick={{fontSize: 9}} stroke="#475569" />
              <YAxis tick={{fontSize: 9}} stroke="#475569" />
              <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px', fontSize: '11px' }} />
              <Legend verticalAlign="top" align="right" wrapperStyle={{ fontSize: '9px', top: -10 }} />
              <Line type="monotone" dataKey="TOC_VU" name="TOC" stroke="#10b981" strokeWidth={1.5} dot={false} isAnimationActive={false} />
              <Line type="monotone" dataKey="TN_VU" name="T-N" stroke="#8b5cf6" strokeWidth={1.5} dot={false} isAnimationActive={false} />
              <Line type="monotone" dataKey="TP_VU" name="T-P" stroke="#f59e0b" strokeWidth={1.5} dot={false} isAnimationActive={false} />
              <Line type="monotone" dataKey="SS_VU" name="SS" stroke="#94a3b8" strokeWidth={1.5} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

    </div>
  );
}