"use client";

import { motion } from "framer-motion";
import { Info } from "lucide-react";

// [추가] 부모(Dashboard)로부터 전달받을 Props 타입 정의
interface Row2Props {
  isDarkMode?: boolean;
}

export default function Row2RiskDetail({ isDarkMode = true }: Row2Props) {
  const riskScore = 72;
  
  const riskFactors = [
    { label: "예상 초과 확률", value: 85, color: "bg-red-500" },
    { label: "기준 대비 여유도(Margin)", value: 40, color: "bg-orange-500" },
    { label: "예상 변화량", value: 65, color: "bg-yellow-500" },
    { label: "데이터 신뢰도", value: 95, color: "bg-emerald-500" },
    { label: "외생요인(강우·계절)", value: 30, color: "bg-blue-500" },
  ];

  // [추가] 테마별 색상 설정
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
        <div>
          <h3 className={`text-md font-bold flex items-center gap-2 transition-colors ${theme.title}`}>
            운영 리스크 점수 상세 <Info className={`w-3.5 h-3.5 transition-colors ${theme.infoIcon}`} />
          </h3>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-2xl font-black text-orange-500">
            {riskScore}<span className={`text-xs font-normal ml-0.5 transition-colors ${theme.scoreSub}`}>/ 100</span>
          </span>
        </div>
      </div>

      {/* 메인 프로그레스 바 */}
      <div className={`relative h-3.5 rounded-full overflow-hidden border transition-colors ${theme.track}`}>
        <motion.div 
          initial={{ width: 0 }}
          animate={{ width: `${riskScore}%` }}
          transition={{ duration: 1 }}
          className="h-full bg-linear-to-r from-orange-600 to-red-600"
        />
      </div>

      {/* 리스크 상세 리스트 (항상 노출) */}
      <div className={`pt-6 space-y-4 mt-4 border-t transition-colors ${theme.divider}`}>
        {riskFactors.map((factor, i) => (
          <div key={i} className="space-y-1.5">
            <div className="flex justify-between text-[11px] font-medium">
              <span className={`transition-colors ${theme.factorLabel}`}>{factor.label}</span>
              <span className={`transition-colors ${theme.factorValue}`}>{factor.value}%</span>
            </div>
            <div className={`h-1 rounded-full overflow-hidden transition-colors ${theme.factorTrack}`}>
              <motion.div 
                initial={{ width: 0 }}
                animate={{ width: `${factor.value}%` }}
                transition={{ delay: i * 0.1 }}
                className={`h-full ${factor.color}`}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}