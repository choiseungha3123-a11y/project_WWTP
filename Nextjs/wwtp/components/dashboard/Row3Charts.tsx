"use client";

import { useMemo } from "react";
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
} from "recharts";

interface TmsRecord {
  SYS_TIME: string;
  TOC_VU: number;
  PH_VU: number;
  SS_VU: number;
  FLUX_VU: number;
  TN_VU: number;
  TP_VU: number;
}

// SWR Fetcher
const fetcher = async (url: string) => {
  const res = await fetch(url);
  if (!res.ok) throw new Error("차트 데이터 로드 실패");
  const json = await res.json();
  return json;
};

export default function Row3Charts() {
  const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL;

  const { data: rawData, error, isLoading } = useSWR(
    `${API_BASE_URL}/api/tmsOrigin/tmsList`,
    fetcher,
    {
      refreshInterval: 30 * 60 * 1000,
      revalidateOnFocus: true,
    }
  );

  // 데이터 추출 및 가공
  const items: TmsRecord[] = useMemo(() => {
    if (rawData?.success && rawData.dataList?.[0]) {
      return rawData.dataList[0];
    }
    return [];
  }, [rawData]);

  const chartData = useMemo(() => {
    return items.map((d) => ({
      ...d,
      displayTime: d.SYS_TIME.split("T")[1]?.substring(0, 5) || d.SYS_TIME,
    }));
  }, [items]);

  const dataDate = useMemo(() => {
    if (items.length > 0 && items[0].SYS_TIME) {
      return items[0].SYS_TIME.split("T")[0];
    }
    return "";
  }, [items]);

  if (isLoading) return <div className="h-full flex items-center justify-center text-slate-500 animate-pulse">차트 데이터 동기화 중...</div>;
  if (error) return <div className="h-full flex items-center justify-center text-red-400 text-xs">차트 데이터를 불러올 수 없습니다.</div>;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 w-full h-full min-h-0">
      
      {/* 1. 유입유량(FLUX) 차트 */}
      <div className="bg-slate-800/40 p-4 rounded-2xl border border-white/5 flex flex-col min-h-0 flex-1">
        <div className="flex justify-between items-center mb-2">
            <h3 className="text-[13px] font-bold text-blue-400">
                유입유량 트렌드 (FLUX)
                <span className="ml-2 text-[10px] text-slate-500 font-normal">{dataDate}</span>
            </h3>
        </div>
        <div className="flex-1 w-full min-h-0"> 
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 5, right: 10, left: -25, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#ffffff05" vertical={false} />
              <XAxis dataKey="displayTime" tick={{fontSize: 9}} stroke="#475569" />
              <YAxis tick={{fontSize: 9}} stroke="#475569" />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px', fontSize: '11px' }}
                itemStyle={{ color: '#fff', padding: '2px 0' }}
              />
              <Legend verticalAlign="top" align="right" wrapperStyle={{ fontSize: '9px', top: -10 }} />
              <Line type="monotone" dataKey="FLUX_VU" name="유량" stroke="#3b82f6" strokeWidth={2} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* 2. 수질 통합 차트 */}
      <div className="bg-slate-800/40 p-4 rounded-2xl border border-white/5 flex flex-col min-h-0 flex-1">
        <div className="flex justify-between items-center mb-2">
             <h3 className="text-[13px] font-bold text-emerald-400">
                수질 통합 분석
                <span className="ml-2 text-[10px] text-slate-500 font-normal">{dataDate}</span>
            </h3>
        </div>
        <div className="flex-1 w-full min-h-0">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 5, right: 10, left: -25, bottom: 5 }}>
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