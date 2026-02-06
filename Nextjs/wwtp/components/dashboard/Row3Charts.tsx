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

  if (loading) return <div className="h-64 flex items-center justify-center text-slate-500">차트 로딩 중...</div>;

  return (
    // min-h-0과 flex-1을 사용하여 공간을 확보합니다.
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 w-full h-full min-h-87.5">
      
      {/* 1. 유입유량(FLUX) 차트 */}
      <div className="bg-slate-800/40 p-4 rounded-2xl border border-white/5 flex flex-col h-87.5">
        <div className="flex justify-between items-center mb-4">
            <h3 className="text-sm font-bold text-blue-400">
                유입유량 트렌드 (FLUX)
                {/* [추가] 날짜 표시 */}
                <span className="ml-2 text-[10px] text-slate-500 font-normal">
                    {dataDate}
                </span>
            </h3>
        </div>
        <div className="flex-1 w-full min-h-0"> 
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#ffffff05" vertical={false} />
              <XAxis dataKey="displayTime" tick={{fontSize: 10}} stroke="#475569" />
              <YAxis tick={{fontSize: 10}} stroke="#475569" />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px', fontSize: '12px' }}
                itemStyle={{ color: '#fff' }}
              />
              <Legend wrapperStyle={{ fontSize: '10px', paddingTop: '10px' }} />
              <Line type="monotone" dataKey="FLUX_VU" name="유량" stroke="#3b82f6" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* 2. 수질 통합 차트 */}
      <div className="bg-slate-800/40 p-4 rounded-2xl border border-white/5 flex flex-col h-87.5">
        <div className="flex justify-between items-center mb-4">
             <h3 className="text-sm font-bold text-emerald-400">
                수질 통합 분석 (TOC/TN/TP/SS)
                 {/* [추가] 날짜 표시 */}
                <span className="ml-2 text-[10px] text-slate-500 font-normal">
                    {dataDate}
                </span>
            </h3>
        </div>
        <div className="flex-1 w-full min-h-0">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#ffffff05" vertical={false} />
              <XAxis dataKey="displayTime" tick={{fontSize: 10}} stroke="#475569" />
              <YAxis tick={{fontSize: 10}} stroke="#475569" />
              <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px', fontSize: '12px' }} />
              <Legend wrapperStyle={{ fontSize: '10px', paddingTop: '10px' }} />
              <Line type="monotone" dataKey="TOC_VU" name="TOC" stroke="#10b981" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="TN_VU" name="T-N" stroke="#8b5cf6" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="TP_VU" name="T-P" stroke="#f59e0b" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="SS_VU" name="SS" stroke="#94a3b8" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

    </div>
  );
}