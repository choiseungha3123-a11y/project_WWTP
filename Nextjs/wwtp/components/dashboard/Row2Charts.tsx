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
  ReferenceLine,
  Legend
} from "recharts";
import { format, parse, parseISO } from "date-fns";

// --- 1. 타입 정의 (JSON 구조에 맞게 보정) ---
interface BoardRecord {
  // 공통 및 실측 키
  SYS_TIME?: string;
  TOC_VU?: number;
  TN_VU?: number;
  TP_VU?: number;
  Q_in?: number;
  // 유입유량 예측 전용 키
  flowTime?: string;
  flowValue?: number;
}

interface ChartPoint {
  timestamp: number;
  displayTime: string;
  value: number; // 유량용
  TOC_VU: number;
  TN_VU: number;
  TP_VU: number;
  isPredicted: boolean;
}

interface BoardViewResponse {
  success: boolean;
  dataList: BoardRecord[][];
}

// --- 2. 시간 파싱 유틸리티 ---
const parseTime = (timeStr: string | undefined): Date | null => {
  if (!timeStr) return null;
  if (timeStr.includes("-") || timeStr.includes("T")) return parseISO(timeStr);
  if (timeStr.length === 14) return parse(timeStr, "yyyyMMddHHmmss", new Date());
  return null;
};

const fetcher = async (url: string) => {
  const token = typeof window !== "undefined" ? localStorage.getItem("accessToken") : "";
  const res = await fetch(url, { headers: { "Authorization": `Bearer ${token || ""}` } });
  return res.json();
};

export default function Row3Charts() {
  const [isClient, setIsClient] = useState(false);
  useEffect(() => setIsClient(true), []);

  const { data: rawData, isLoading } = useSWR<BoardViewResponse>(
    isClient ? `${process.env.NEXT_PUBLIC_API_URL}/api/board/boardView` : null,
    fetcher
  );

  // --- 3. 핵심 가공 로직 ---
  const processedData = useMemo(() => {
    if (!rawData?.success || !rawData.dataList) return { flowData: [], waterData: [], flowRef: null, waterRef: null };

    // A. 유입유량 가공 (실측: Q_in / 예측: flowValue)
    const rawFlowList = rawData.dataList.find(l => l.length > 0 && ('Q_in' in l[0] || 'flowValue' in l[0])) || [];
    const flowPoints: ChartPoint[] = rawFlowList.map(item => {
      const isPred = 'flowValue' in item; // flowValue 키가 있으면 예측
      const tStr = isPred ? item.flowTime : item.SYS_TIME;
      const d = parseTime(tStr) || new Date();
      return {
        timestamp: d.getTime(),
        displayTime: format(d, "MM-dd HH:mm"),
        value: (isPred ? item.flowValue : item.Q_in) ?? 0,
        TOC_VU: 0, TN_VU: 0, TP_VU: 0, // 유량차트엔 필요없음
        isPredicted: isPred
      };
    }).sort((a, b) => a.timestamp - b.timestamp);

    // B. 수질 가공 (실측: 14자리 SYS_TIME / 예측: ISO SYS_TIME)
    const rawWaterList = rawData.dataList.find(l => l.length > 0 && 'TOC_VU' in l[0]) || [];
    const waterPoints: ChartPoint[] = rawWaterList.map(item => {
      const tStr = item.SYS_TIME || "";
      const isPred = tStr.includes("-"); // ISO 포맷이면 예측
      const d = parseTime(tStr) || new Date();
      return {
        timestamp: d.getTime(),
        displayTime: format(d, "MM-dd HH:mm"),
        value: 0,
        TOC_VU: item.TOC_VU ?? 0,
        TN_VU: item.TN_VU ?? 0,
        TP_VU: item.TP_VU ?? 0,
        isPredicted: isPred
      };
    }).sort((a, b) => a.timestamp - b.timestamp);

    // 구분선(ReferenceLine) 위치 찾기
    const fRef = flowPoints.find(p => p.isPredicted)?.timestamp || null;
    const wRef = waterPoints.find(p => p.isPredicted)?.timestamp || null;

    return { flowData: flowPoints, waterData: waterPoints, flowRef: fRef, waterRef: wRef };
  }, [rawData]);

  if (isLoading) return <div className="h-full flex items-center justify-center text-slate-400">데이터 수신 중...</div>;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 w-full h-full min-h-0 p-2">
      
      {/* 유입유량 차트 */}
      <div className="bg-slate-900/50 p-4 rounded-xl border border-white/5 flex flex-col">
        <h3 className="text-sm font-bold text-blue-400 mb-4">유입유량 실측 + 예측</h3>
        <div className="flex-1 min-h-0">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={processedData.flowData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
              <XAxis 
                dataKey="timestamp" 
                type="number" 
                domain={['dataMin', 'dataMax']} 
                scale="time"
                tickFormatter={(t) => format(t, "HH:mm")}
                stroke="#64748b" fontSize={10}
              />
              <YAxis stroke="#64748b" fontSize={10} />
              <Tooltip labelFormatter={(t) => format(t, "MM-dd HH:mm")} />
              <Legend verticalAlign="top" align="right" iconType="circle" />
              {processedData.flowRef && (
                <ReferenceLine x={processedData.flowRef} stroke="#ef4444" strokeDasharray="5 5" label={{ value: 'NOW', fill: '#ef4444', fontSize: 12 }} />
              )}
              <Line 
                type="monotone" 
                dataKey="value" 
                name="유량" 
                stroke="#3b82f6" 
                strokeWidth={2} 
                dot={false}
                // 예측 구간(30분 단위)은 점을 찍어 구분
                activeDot={{ r: 4 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* 수질 차트 */}
      <div className="bg-slate-900/50 p-4 rounded-xl border border-white/5 flex flex-col">
        <h3 className="text-sm font-bold text-emerald-400 mb-4">수질 통합 예측</h3>
        <div className="flex-1 min-h-0">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={processedData.waterData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
              <XAxis 
                dataKey="timestamp" 
                type="number" 
                domain={['dataMin', 'dataMax']} 
                scale="time"
                tickFormatter={(t) => format(t, "HH:mm")}
                stroke="#64748b" fontSize={10}
              />
              <YAxis stroke="#64748b" fontSize={10} />
              <Tooltip labelFormatter={(t) => format(t, "MM-dd HH:mm")} />
              <Legend verticalAlign="top" align="right" />
              {processedData.waterRef && (
                <ReferenceLine x={processedData.waterRef} stroke="#ef4444" strokeDasharray="5 5" label={{ value: 'NOW', fill: '#ef4444', fontSize: 12 }} />
              )}
              <Line type="monotone" dataKey="TOC_VU" name="TOC" stroke="#10b981" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="TN_VU" name="T-N" stroke="#8b5cf6" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="TP_VU" name="T-P" stroke="#f59e0b" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

    </div>
  );
}