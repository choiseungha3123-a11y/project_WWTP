"use client";

import { useMemo, useSyncExternalStore } from "react";
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
}

interface MergedData {
  displayTime: string;
  fullTime: string; 
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

interface Row1Props {
  isDarkMode?: boolean;
}

interface PulsingDotProps {
  cx?: number;
  cy?: number;
  stroke?: string;
  payload?: MergedData;
  lastFullTime: string;
}

type StatusType = "danger" | "warning" | "normal";

// ----------------------------------------------------------------------
// 2. 유틸리티 함수 및 상태 판별 로직
// ----------------------------------------------------------------------

const fetcher = async (url: string): Promise<BoardViewResponse> => {
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

const formatDisplayTime = (timeStr: string): string => {
  if (!timeStr) return "";
  if (timeStr.includes("T")) return timeStr.split("T")[1].substring(0, 5);
  if (timeStr.length >= 12) return `${timeStr.substring(8, 10)}:${timeStr.substring(10, 12)}`;
  return timeStr;
};

const formatFullTime = (timeStr: string): string => {
  if (!timeStr) return "";
  return timeStr.replace(/[-T:]/g, "").substring(0, 12);
};

// 실시간 상태 판별 함수
const getStatus = (title: string, val: number | undefined): StatusType => {
  if (val === undefined || val === null) return "normal";
  
  // pH 특수 로직
  if (title === "pH") {
    if (val <= 5.8 || val >= 8.5) return "danger";
    if (val <= 6.0 || val >= 8.3) return "warning";
    return "normal";
  }
  
  // 수질 지표 기준값
  const limits: Record<string, number> = { 
    "TOC": 15, 
    "T-N": 20, 
    "T-P": 0.5, 
    "SS": 10 
  };
  
  const limit = limits[title];
  if (!limit) return "normal";

  if (val >= limit) return "danger";
  if (val >= limit * 0.9) return "warning";
  return "normal";
};

const subscribe = () => () => {}; 
const getSnapshot = () => true;   
const getServerSnapshot = () => false;

const PulsingDot = (props: PulsingDotProps) => {
  const { cx, cy, stroke, payload, lastFullTime } = props;
  if (!cx || !cy || !payload || payload.fullTime !== lastFullTime) return null;
  return (
    <g>
      <circle cx={cx} cy={cy} r={6} fill={stroke} opacity="0.6">
        <animate attributeName="r" from="6" to="14" dur="1.8s" begin="0s" repeatCount="indefinite" />
        <animate attributeName="opacity" from="0.6" to="0" dur="1.8s" begin="0s" repeatCount="indefinite" />
      </circle>
      <circle cx={cx} cy={cy} r={4.5} fill={stroke} stroke="#ffffff" strokeWidth={2} style={{ filter: "drop-shadow(0px 0px 3px rgba(0,0,0,0.5))" }} />
    </g>
  );
};

// ----------------------------------------------------------------------
// 3. 메인 컴포넌트
// ----------------------------------------------------------------------

export default function Row1Charts({ isDarkMode = true }: Row1Props) {
  const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL;
  const isClient = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);

  const { data: rawData, error, isLoading } = useSWR<BoardViewResponse>(
    isClient ? `${API_BASE_URL}/api/board/boardView` : null,
    fetcher,
    { refreshInterval: 30 * 60 * 1000 }
  );

  const { chartData, latestValues, lastActualTime } = useMemo(() => {
    if (!rawData?.success || !rawData.dataList || rawData.dataList.length < 4) {
      return { chartData: [] as MergedData[], latestValues: null as LatestValues | null, lastActualTime: "" };
    }

    const [actualTms, predictTms, actualFlow, predictFlow] = rawData.dataList;
    const mergedMap = new Map<string, MergedData>();

    const processTmsItems = (items: TmsRecord[], isActual: boolean) => {
      items.forEach((item) => {
        const fullTime = formatFullTime(item.SYS_TIME);
        const displayTime = formatDisplayTime(item.SYS_TIME);
        const existing = mergedMap.get(fullTime) || { displayTime, fullTime };
        if (isActual) {
          existing.toc_A = item.TOC_VU; existing.ph_A = item.PH_VU; existing.ss_A = item.SS_VU;
          existing.flux_A = item.FLUX_VU; existing.tn_A = item.TN_VU; existing.tp_A = item.TP_VU;
        } else {
          existing.toc_P = item.TOC_VU; existing.ph_P = item.PH_VU; existing.ss_P = item.SS_VU;
          existing.flux_P = item.FLUX_VU; existing.tn_P = item.TN_VU; existing.tp_P = item.TP_VU;
        }
        mergedMap.set(fullTime, existing);
      });
    };

    const processFlowItems = (items: FlowRecord[], isActual: boolean) => {
      items.forEach((item) => {
        const fullTime = formatFullTime(item.SYS_TIME);
        const displayTime = formatDisplayTime(item.SYS_TIME);
        const existing = mergedMap.get(fullTime) || { displayTime, fullTime };
        if (isActual) existing.Q_in_A = item.Q_in;
        else existing.Q_in_P = item.Q_in;
        mergedMap.set(fullTime, existing);
      });
    };

    processTmsItems(actualTms, true);
    processTmsItems(predictTms, false);
    processFlowItems(actualFlow, true);
    processFlowItems(predictFlow, false);

    const sortedData = Array.from(mergedMap.values()).sort((a, b) => a.fullTime.localeCompare(b.fullTime));
    const lastActualRecord = actualTms.length > 0 ? actualTms[actualTms.length - 1] : null;
    const lastActualTimeStr = lastActualRecord ? formatFullTime(lastActualRecord.SYS_TIME) : "";
    const lastFlow = actualFlow.length > 0 ? actualFlow[actualFlow.length - 1] : null;

    return { 
      chartData: sortedData, 
      latestValues: lastActualRecord ? { ...lastActualRecord, Q_in: lastFlow?.Q_in } : null, 
      lastActualTime: lastActualTimeStr 
    };
  }, [rawData]);

  const themeColors = {
    grid: isDarkMode ? "#ffffff08" : "#e2e8f0",
    axis: isDarkMode ? "#475569" : "#94a3b8",
    tick: isDarkMode ? "#94a3b8" : "#64748b",
    tooltipBg: isDarkMode ? "#1e293b" : "#ffffff",
    tooltipBorder: isDarkMode ? "#334155" : "#e2e8f0",
    tooltipText: isDarkMode ? "#f8fafc" : "#1e293b"
  };

  const renderRow = (
    title: string, 
    color: string, 
    keys: { actual: string; predict: string }, 
    latestVal: number | undefined, 
    unit: string = ""
  ) => {
    const status = getStatus(title, latestVal);
    
    // 상태별 동적 스타일 설정
    const getStatusStyles = () => {
      if (status === "danger") return {
        cardBg: isDarkMode ? "bg-red-900/40" : "bg-red-50",
        border: "border-red-500/50",
        label: isDarkMode ? "text-red-200" : "text-red-700",
        valColor: isDarkMode ? "text-red-400" : "text-red-600",
        chartBg: isDarkMode ? "bg-red-950/20" : "bg-red-50/30",
        lineColor: "#ef4444"
      };
      if (status === "warning") return {
        cardBg: isDarkMode ? "bg-amber-900/30" : "bg-amber-50",
        border: "border-amber-500/40",
        label: isDarkMode ? "text-amber-200" : "text-amber-700",
        valColor: isDarkMode ? "text-amber-400" : "text-amber-600",
        chartBg: isDarkMode ? "bg-amber-950/10" : "bg-amber-50/20",
        lineColor: "#f59e0b"
      };
      return {
        cardBg: isDarkMode ? "bg-slate-700/80" : "bg-white",
        border: isDarkMode ? "border-white/10" : "border-blue-100",
        label: isDarkMode ? "text-slate-200" : "text-blue-900",
        valColor: color, // Normal 상태에선 고유 색상 유지
        chartBg: isDarkMode ? "bg-slate-800/40" : "bg-slate-50/50",
        lineColor: color
      };
    };

    const s = getStatusStyles();

    return (
      <div className="flex w-full items-stretch mb-5 last:mb-0 h-24">
        {/* 왼쪽 정보 박스 */}
        <div 
          className={`w-32 rounded-l-2xl border-y border-l flex flex-col justify-center items-center p-3 shrink-0 shadow-lg transition-all duration-500 ${s.cardBg} ${s.border}`}
          style={{ borderLeft: `4px solid ${s.lineColor}` }}
        >
          <span className={`text-xl font-bold mb-1 uppercase tracking-wider ${s.label}`}>{title}</span>
          <span className={`text-xl font-black tracking-tighter drop-shadow-md ${s.valColor}`}>
            {latestVal !== undefined && latestVal !== null ? latestVal.toFixed(2) : "-"}
          </span>
          {unit && <span className={`text-[10px] mt-1 font-semibold ${isDarkMode ? "text-slate-400" : "text-slate-500"}`}>{unit}</span>}
        </div>

        {/* 오른쪽 차트 박스 */}
        <div className={`flex-1 ml-1 rounded-r-2xl border p-2 overflow-hidden shadow-sm transition-all duration-500 ${s.chartBg} ${s.border}`}>
          <ResponsiveContainer width="100%" height={100}>
            <LineChart data={chartData} margin={{ top: 5, right: 10, left: -30, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={themeColors.grid} vertical={false} />
              <XAxis 
                dataKey="fullTime" 
                tick={{ fontSize: 9, fill: themeColors.tick }} 
                stroke={themeColors.axis} 
                interval={5} 
                tickFormatter={(value) => (value && value.length >= 12) ? `${value.substring(8, 10)}:${value.substring(10, 12)}` : value}
              />
              <YAxis tick={{ fontSize: 9, fill: themeColors.tick }} stroke={themeColors.axis} domain={['auto', 'auto']} />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: themeColors.tooltipBg, 
                  border: `1px solid ${themeColors.tooltipBorder}`, 
                  fontSize: '11px', 
                  borderRadius: '8px', 
                  color: themeColors.tooltipText 
                }} 
                labelFormatter={(value) => (value && value.length >= 12) ? `${value.substring(8, 10)}:${value.substring(10, 12)}` : value}
              />
              <ReferenceLine x={lastActualTime} stroke={isDarkMode ? "#ffffff50" : "#00000030"} strokeDasharray="3 3" />
              
              <Line 
                type="monotone" 
                dataKey={keys.actual} 
                stroke={s.lineColor} 
                strokeWidth={3} 
                dot={<PulsingDot lastFullTime={lastActualTime} />} 
                connectNulls 
                isAnimationActive={false} 
              />
              
              <Line 
                type="monotone" 
                dataKey={keys.predict} 
                stroke={s.lineColor} 
                strokeWidth={2} 
                strokeDasharray="4 4" 
                dot={false} 
                connectNulls 
                isAnimationActive={false} 
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    );
  };

  if (isLoading || !isClient) return <div className="p-4 text-slate-500">Loading...</div>;
  if (error) return <div className="p-4 text-red-400">Error!</div>;

  return (
    <div className="flex flex-col w-full h-full p-4 overflow-y-auto custom-scrollbar">
      {renderRow("유입유량", "#ef4444", { actual: "Q_in_A", predict: "Q_in_P" }, latestValues?.Q_in, "m³/hr")}
      {renderRow("FLUX", "#f97316", { actual: "flux_A", predict: "flux_P" }, latestValues?.FLUX_VU, "m³/hr")}
      {renderRow("pH", "#facc15", { actual: "ph_A", predict: "ph_P" }, latestValues?.PH_VU)}
      {renderRow("SS", "#10b981", { actual: "ss_A", predict: "ss_P" }, latestValues?.SS_VU, "mg/L")}
      {renderRow("TOC", "#3b82f6", { actual: "toc_A", predict: "toc_P" }, latestValues?.TOC_VU, "mg/L")}
      {renderRow("T-N", "#6366f1", { actual: "tn_A", predict: "tn_P" }, latestValues?.TN_VU, "mg/L")}
      {renderRow("T-P", "#a855f7", { actual: "tp_A", predict: "tp_P" }, latestValues?.TP_VU, "mg/L")}
    </div>
  );
}