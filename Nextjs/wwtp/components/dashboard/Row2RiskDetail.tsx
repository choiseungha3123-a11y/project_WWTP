"use client";

import { useMemo, useSyncExternalStore } from "react";
import useSWR from "swr";
import { motion } from "framer-motion";
import { Info } from "lucide-react";

// ----------------------------------------------------------------------
// 1. 인터페이스 및 상수 정의
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
interface FlowRecord { SYS_TIME: string; Q_in: number; }
interface WeatherData { SYS_TIME: string; TA: number; RN_15m: number; HM: number; }

interface BoardViewResponse {
  success: boolean;
  dataList: [TmsRecord[], TmsRecord[], FlowRecord[], FlowRecord[], WeatherData[]]; 
}

interface Row2Props { isDarkMode?: boolean; latestWeather: WeatherData | null; }

const LIMITS = { pH: 8.6, SS: 10, TOC: 15, TN: 20, TP: 2 };
const MODEL_R2 = { pH: 0.92, SS: 0.88, TOC: 0.95, TN: 0.90, TP: 0.85 };

const WEIGHTS = {
  probExceed: 0.2,
  margin: 0.4,
  expectedChange: 0.15,
  reliability: 0.15,
  external: 0.1
};

const fetcher = async (url: string): Promise<BoardViewResponse> => {
  const token = typeof window !== "undefined" ? localStorage.getItem("accessToken") : null;
  const res = await fetch(url, {
    headers: { Authorization: token ? `Bearer ${token}` : "", "Content-Type": "application/json" },
  });
  if (!res.ok) throw new Error("데이터 로드 실패");
  return res.json();
};

const subscribe = () => () => {}; 
const getSnapshot = () => true;   
const getServerSnapshot = () => false;

// ----------------------------------------------------------------------
// 2. 메인 컴포넌트
// ----------------------------------------------------------------------
export default function Row2RiskDetail({ isDarkMode = true, latestWeather }: Row2Props) {
  const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL;
  const isClient = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);

  const { data: rawData } = useSWR<BoardViewResponse>(
    isClient ? `${API_BASE_URL}/api/board/boardView` : null,
    fetcher,
    { refreshInterval: 30000 }
  );

  const computedFactors = useMemo(() => {
    let probExceedVal = 100;      
    let marginScoreVal = 100;      
    let expectedChangeVal = 100; 
    let reliabilityVal = 90;      
    let externalScoreVal = 100;  

    if (rawData?.success && rawData.dataList) {
      const [actualTms, predictTms, actualFlow, predictFlow] = rawData.dataList;
      const latestActual = actualTms[actualTms.length - 1];
      const latestPredict = predictTms[predictTms.length - 1];

      // --- [항목 1] 예상 초과 확률 (유지) ---
      if (latestPredict) {
        let exceedCount = 0;
        if (latestPredict.PH_VU > LIMITS.pH) exceedCount++;
        if (latestPredict.SS_VU > LIMITS.SS) exceedCount++;
        if (latestPredict.TOC_VU > LIMITS.TOC) exceedCount++;
        if (latestPredict.TN_VU > LIMITS.TN) exceedCount++;
        if (latestPredict.TP_VU > LIMITS.TP) exceedCount++;
        probExceedVal = Math.max(0, 100 - (20 * exceedCount));
      }

      // --- [항목 2] 기준 대비 여유도 (기존 방식 유지) ---
      if (latestActual) {
        const proximityScores = [
          (latestActual.PH_VU / LIMITS.pH),
          (latestActual.SS_VU / LIMITS.SS),
          (latestActual.TOC_VU / LIMITS.TOC),
          (latestActual.TN_VU / LIMITS.TN),
          (latestActual.TP_VU / LIMITS.TP)
        ];
        const avgProximity = (proximityScores.reduce((a, b) => a + b, 0) / proximityScores.length) * 100;
        marginScoreVal = Math.max(0, 100 - avgProximity);
      }

      // --- [항목 3] 예상 변화량 (초강력 감점 수정본) ---
      if (actualTms.length > 0 && predictTms.length > 0) {
        let totalScoreSum = 0;
        let matchedCount = 0;
        const sampleCount = Math.min(actualTms.length, 7);

        for (let i = 1; i <= sampleCount; i++) {
          const act = actualTms[actualTms.length - i];
          const actTime = new Date(act.SYS_TIME).getTime();
          
          // 매칭 시간 범위를 20분으로 확대하여 데이터 유실 대응
          const pre = predictTms.find(p => Math.abs(new Date(p.SYS_TIME).getTime() - actTime) < 20 * 60 * 1000);
          const actF = actualFlow?.find(f => Math.abs(new Date(f.SYS_TIME).getTime() - actTime) < 20 * 60 * 1000);
          const preF = predictFlow?.find(f => Math.abs(new Date(f.SYS_TIME).getTime() - actTime) < 20 * 60 * 1000);

          if (pre) {
            const errors = [
              Math.abs(pre.PH_VU - act.PH_VU) / Math.max(act.PH_VU, 0.1),
              Math.abs(pre.TOC_VU - act.TOC_VU) / Math.max(act.TOC_VU, 0.1),
              Math.abs(pre.SS_VU - act.SS_VU) / Math.max(act.SS_VU, 0.1),
              Math.abs(pre.TN_VU - act.TN_VU) / Math.max(act.TN_VU, 0.1),
              Math.abs(pre.TP_VU - act.TP_VU) / Math.max(act.TP_VU, 0.01),
              // [추가] Flux 오차 계산 (5012 vs 9336 대응)
              Math.abs(pre.FLUX_VU - act.FLUX_VU) / Math.max(act.FLUX_VU, 1)
            ];
            
            if (actF && preF) {
              // [추가] 유입유량 오차 계산
              errors.push(Math.abs(preF.Q_in - actF.Q_in) / Math.max(actF.Q_in, 1));
            }

            // 오차 10%당 40점 감점 (민감도 상향)
            const itemScores = errors.map(err => 100 - (err * 400));
            
            // 최악의 오차를 보인 항목을 90% 반영하여 점수 희석 방지
            const minItem = Math.min(...itemScores);
            const avgItem = itemScores.reduce((a, b) => a + b, 0) / itemScores.length;
            
            totalScoreSum += Math.max(0, (minItem * 0.9 + avgItem * 0.1));
            matchedCount++;
          }
        }
        if (matchedCount > 0) expectedChangeVal = totalScoreSum / matchedCount;
      }
    }

    // --- [항목 4] 데이터 신뢰도 (유지) ---
    reliabilityVal = (Object.values(MODEL_R2).reduce((a, b) => a + b, 0) / 5) * 100;

    // --- [항목 5] 외생요인 (유지) ---
    const rainValue = latestWeather?.RN_15m || 0;
    externalScoreVal = Math.max(0, 100 - (rainValue * 10));

    return [
      { id: 'probExceed', label: "예상 초과 확률", value: Math.round(probExceedVal), weight: WEIGHTS.probExceed, color: "bg-red-500" },
      { id: 'margin', label: "기준 대비 여유도(Margin)", value: Math.round(marginScoreVal), weight: WEIGHTS.margin, color: "bg-orange-500" },
      { id: 'expectedChange', label: "예상 변화량", value: Math.round(expectedChangeVal), weight: WEIGHTS.expectedChange, color: "bg-yellow-500" },
      { id: 'reliability', label: "데이터 신뢰도", value: Math.round(reliabilityVal), weight: WEIGHTS.reliability, color: "bg-emerald-500" },
      { id: 'external', label: "외생요인(강우·계절)", value: Math.round(externalScoreVal), weight: WEIGHTS.external, color: "bg-blue-500" },
    ];
  }, [rawData, latestWeather]);

  const riskScore = useMemo(() => {
    const weightedSum = computedFactors.reduce((acc, factor) => acc + (factor.value * factor.weight), 0);
    return Math.round(weightedSum);
  }, [computedFactors]);

  // 테마 및 렌더링 로직 (기존과 동일)
  const theme = {
    container: isDarkMode ? "bg-slate-800/40 border-white/10" : "bg-white border-blue-100",
    title: isDarkMode ? "text-white" : "text-slate-900",
    infoIcon: isDarkMode ? "text-slate-500" : "text-blue-400",
    scoreSub: isDarkMode ? "text-slate-500" : "text-slate-400",
    track: isDarkMode ? "bg-slate-900/50 border-white/5" : "bg-blue-50 border-blue-100",
    divider: isDarkMode ? "border-white/5" : "border-slate-100",
    factorLabel: isDarkMode ? "text-slate-400" : "text-slate-500",
    factorValue: isDarkMode ? "text-slate-200" : "text-slate-700",
    factorTrack: isDarkMode ? "bg-slate-900" : "bg-slate-100",
  };

  return (
    <div className={`p-5 rounded-3xl border shadow-xl select-none transition-colors duration-500 ${theme.container}`}>
      <div className="flex justify-between items-center mb-4">
        <h3 className={`text-md font-bold flex items-center gap-2 transition-colors ${theme.title}`}>
          운영 리스크 점수 상세 <Info className={`w-3.5 h-3.5 transition-colors ${theme.infoIcon}`} />
        </h3>
        <div className="flex items-center gap-3">
          <span className="text-2xl font-black text-orange-500">
            {riskScore}<span className={`text-xs font-normal ml-0.5 transition-colors ${theme.scoreSub}`}>/ 100</span>
          </span>
        </div>
      </div>

      <div className={`relative h-3.5 rounded-full overflow-hidden border transition-colors ${theme.track}`}>
        <motion.div 
          key={riskScore}
          initial={{ width: 0 }}
          animate={{ width: `${riskScore}%` }}
          transition={{ duration: 1, ease: "easeOut" }}
          className="h-full bg-linear-to-r from-orange-600 to-red-600"
        />
      </div>

      <div className={`pt-6 space-y-4 mt-4 border-t transition-colors ${theme.divider}`}>
        {computedFactors.map((factor, i) => (
          <div key={factor.id} className="space-y-1.5">
            <div className="flex justify-between text-[11px] font-medium">
              <div className="flex items-center gap-1.5">
                <span className={`transition-colors ${theme.factorLabel}`}>{factor.label}</span>
                <span className="text-[9px] opacity-40 font-light">w:{(factor.weight).toFixed(2)}</span>
              </div>
              <span className={`transition-colors ${theme.factorValue}`}>{factor.value}%</span>
            </div>
            <div className={`h-1 rounded-full overflow-hidden transition-colors ${theme.factorTrack}`}>
              <motion.div 
                initial={{ width: 0 }}
                animate={{ width: `${factor.value}%` }}
                transition={{ delay: i * 0.1, duration: 0.8 }}
                className={`h-full ${factor.color}`}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}