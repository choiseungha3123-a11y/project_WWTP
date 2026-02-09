"use client";

import { useEffect, useMemo, useState } from "react";
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

export default function Row3Charts() {
  const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL;

  const [items, setItems] = useState<TmsRecord[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/api/tmsOrigin/tmsList`);
        const json = await res.json();
        if (json.success && json.dataList[0]) {
          setItems(json.dataList[0]);
        }
      } catch (e) {
        console.error("차트 데이터 로드 실패:", e);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  const chartData = useMemo(() => {
    return items.map((d) => ({
      ...d,
      displayTime: d.SYS_TIME.split("T")[1]?.substring(0, 5) || d.SYS_TIME,
    }));
  }, [items]);

  // [추가] 데이터의 날짜 추출 (YYYY-MM-DD)
  // 데이터가 있다면 첫 번째 데이터의 날짜를 기준 날짜로 사용
  const dataDate = useMemo(() => {
    if (items.length > 0 && items[0].SYS_TIME) {
        return items[0].SYS_TIME.split("T")[0]; 
    }
    return "";
  }, [items]);

  if (loading) return <div className="h-full flex items-center justify-center text-slate-500">차트 로딩 중...</div>;

  return (
    // h-full과 flex-1을 주어 부모가 주는 공간을 꽉 채우게 합니다.
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 w-full h-full min-h-0">
      
      {/* 1. 유입유량(FLUX) 차트 */}
      {/* 고정 h-87.5 제거, flex-1 추가 */}
      <div className="bg-slate-800/40 p-4 rounded-2xl border border-white/5 flex flex-col min-h-0 flex-1">
        <div className="flex justify-between items-center mb-2">
            <h3 className="text-[13px] font-bold text-blue-400">
                유입유량 트렌드 (FLUX)
                <span className="ml-2 text-[10px] text-slate-500 font-normal">{dataDate}</span>
            </h3>
        </div>
        <div className="flex-1 w-full min-h-0"> 
          <ResponsiveContainer width="100%" height="100%">
            {/* bottom: 0 -> 5로 조정하여 x축 라벨 공간 확보 */}
            <LineChart data={chartData} margin={{ top: 5, right: 10, left: -25, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#ffffff05" vertical={false} />
              <XAxis dataKey="displayTime" tick={{fontSize: 9}} stroke="#475569" />
              <YAxis tick={{fontSize: 9}} stroke="#475569" />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px', fontSize: '11px' }}
                itemStyle={{ color: '#fff', padding: '2px 0' }}
              />
              {/* Legend의 paddingTop을 줄여 차트 영역 확보 */}
              <Legend verticalAlign="top" align="right" wrapperStyle={{ fontSize: '9px', top: -10 }} />
              <Line type="monotone" dataKey="FLUX_VU" name="유량" stroke="#3b82f6" strokeWidth={2} dot={false} />
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
              <Line type="monotone" dataKey="TOC_VU" name="TOC" stroke="#10b981" strokeWidth={1.5} dot={false} />
              <Line type="monotone" dataKey="TN_VU" name="T-N" stroke="#8b5cf6" strokeWidth={1.5} dot={false} />
              <Line type="monotone" dataKey="TP_VU" name="T-P" stroke="#f59e0b" strokeWidth={1.5} dot={false} />
              <Line type="monotone" dataKey="SS_VU" name="SS" stroke="#94a3b8" strokeWidth={1.5} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

    </div>
  );
}