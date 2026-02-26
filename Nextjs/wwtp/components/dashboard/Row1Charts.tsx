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

// ----------------------------------------------------------------------
// 2. 유틸리티 함수 및 스토어 설정
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

const subscribe = () => () => {}; 
const getSnapshot = () => true;   
const getServerSnapshot = () => false;

// ----------------------------------------------------------------------
// [수정] shadow 오류 해결 및 타입 안전성 확보
// ----------------------------------------------------------------------
const PulsingDot = (props: PulsingDotProps) => {
  const { cx, cy, stroke, payload, lastFullTime } = props;

  if (!cx || !cy || !payload || payload.fullTime !== lastFullTime) return null;

  return (
    <g>
      {/* 바깥쪽 퍼지는 애니메이션 */}
      <circle cx={cx} cy={cy} r={6} fill={stroke} opacity="0.6">
        <animate
          attributeName="r"
          from="6"
          to="14"
          dur="1.8s"
          begin="0s"
          repeatCount="indefinite"
        />
        <animate
          attributeName="opacity"
          from="0.6"
          to="0"
          dur="1.8s"
          begin="0s"
          repeatCount="indefinite"
        />
      </circle>
      
      {/* 중심점: shadow 속성 대신 style의 filter를 사용 */}
      <circle 
        cx={cx} 
        cy={cy} 
        r={4.5} 
        fill={stroke} 
        stroke="#ffffff" 
        strokeWidth={2} 
        style={{ filter: "drop-shadow(0px 0px 3px rgba(0,0,0,0.5))" }}
      />
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

    const sortedData = Array.from(mergedMap.values()).sort((a, b) => 
      a.fullTime.localeCompare(b.fullTime)
    );
    
    const lastActualRecord = actualTms.length > 0 ? actualTms[actualTms.length - 1] : null;
    const lastActualTimeStr = lastActualRecord ? formatFullTime(lastActualRecord.SYS_TIME) : "";
    const lastFlow = actualFlow.length > 0 ? actualFlow[actualFlow.length - 1] : null;

    const latest: LatestValues | null = lastActualRecord ? {
      ...lastActualRecord,
      Q_in: lastFlow?.Q_in
    } : null;

    return { chartData: sortedData, latestValues: latest, lastActualTime: lastActualTimeStr };
  }, [rawData]);

  const themeColors = {
    cardBg: isDarkMode ? "bg-slate-700/80" : "bg-white",
    chartBoxBg: isDarkMode ? "bg-slate-800/40" : "bg-slate-50/50",
    border: isDarkMode ? "border-white/10" : "border-blue-100",
    label: isDarkMode ? "text-slate-200" : "text-blue-900",
    unit: isDarkMode ? "text-slate-300" : "text-slate-500",
    grid: isDarkMode ? "#ffffff08" : "#e2e8f0",
    axis: isDarkMode ? "#475569" : "#94a3b8",
    tick: isDarkMode ? "#94a3b8" : "#64748b",
    tooltipBg: isDarkMode ? "#1e293b" : "#ffffff",
    tooltipBorder: isDarkMode ? "#334155" : "#e2e8f0",
    tooltipText: isDarkMode ? "#f8fafc" : "#1e293b"
  };

  if (isLoading || !isClient) return <div className="p-4 text-slate-500">Loading...</div>;
  if (error) return <div className="p-4 text-red-400">Error!</div>;

  const renderRow = (
    title: string, 
    color: string, 
    keys: { actual: string; predict: string }, 
    latestVal: number | undefined, 
    unit: string = ""
  ) => (
    <div className="flex w-full items-stretch mb-5 last:mb-0 h-24">
      <div 
        className={`w-32 rounded-l-2xl border-y border-l flex flex-col justify-center items-center p-3 shrink-0 shadow-lg transition-colors duration-500 ${themeColors.cardBg} ${themeColors.border}`}
        style={{ borderLeft: `4px solid ${color}` }}
      >
        <span className={`text-xl font-bold mb-1 uppercase tracking-wider ${themeColors.label}`}>{title}</span>
        <span className="text-xl font-black tracking-tighter drop-shadow-md" style={{ color }}>
          {latestVal !== undefined && latestVal !== null ? latestVal.toFixed(2) : "-"}
        </span>
        {unit && <span className={`text-[10px] mt-1 font-semibold ${themeColors.unit}`}>{unit}</span>}
      </div>

      <div className={`flex-1 ml-1 rounded-r-2xl border p-2 overflow-hidden shadow-sm transition-colors duration-500 ${themeColors.chartBoxBg} ${themeColors.border}`}>
        <ResponsiveContainer width="100%" height={100}>
          <LineChart data={chartData} margin={{ top: 5, right: 10, left: -30, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={themeColors.grid} vertical={false} />
            <XAxis 
              dataKey="fullTime" 
              tick={{ fontSize: 9, fill: themeColors.tick }} 
              stroke={themeColors.axis} 
              interval={5} 
              tickFormatter={(value) => {
                if (value && value.length >= 12) {
                  return `${value.substring(8, 10)}:${value.substring(10, 12)}`;
                }
                return value;
              }}
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
              itemStyle={{ fontWeight: 'bold' }}
              labelFormatter={(value) => {
                if (value && value.length >= 12) {
                  return `${value.substring(8, 10)}:${value.substring(10, 12)}`;
                }
                return value;
              }}
            />
            <ReferenceLine 
              x={lastActualTime} 
              stroke={isDarkMode ? "#ffffff50" : "#00000030"} 
              strokeDasharray="3 3" 
            />
            
            <Line 
              type="monotone" 
              dataKey={keys.actual} 
              stroke={color} 
              strokeWidth={3} 
              dot={<PulsingDot lastFullTime={lastActualTime} />} 
              connectNulls 
              isAnimationActive={false} 
            />
            
            <Line 
              type="monotone" 
              dataKey={keys.predict} 
              stroke={color} 
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

  return (
    <div className="flex flex-col w-full h-full p-4 overflow-y-auto">
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