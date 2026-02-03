"use client";

import { motion } from "framer-motion";
import { Info, AlertCircle } from "lucide-react";

export default function Row4RiskDetail() {
  // 샘플 데이터 (실제로는 API나 상위 props에서 전달)
  const riskScore = 72;
  const riskFactors = [
    { label: "기준 초과 확률", value: 85, weight: "high", color: "bg-red-500" },
    { label: "기준 대비 여유도(Margin)", value: 40, weight: "medium", color: "bg-orange-500" },
    { label: "변화 속도(급변성)", value: 65, weight: "medium", color: "bg-yellow-500" },
    { label: "데이터 신뢰도(결측·이상)", value: 95, weight: "low", color: "bg-emerald-500" },
    { label: "외생요인(강우·계절)", value: 30, weight: "low", color: "bg-blue-500" },
  ];

  return (
    <div className="bg-slate-800/50 p-6 rounded-3xl border border-white/10 h-full shadow-xl">
      <div className="flex justify-between items-start mb-6">
        <div>
          <h3 className="text-lg font-bold text-white flex items-center gap-2">
            운영 리스크 점수 상세 <Info className="w-4 h-4 text-slate-400" />
          </h3>
          <p className="text-xs text-slate-400 mt-1">종합 판단 근거를 관리자에게 설명</p>
        </div>
        <div className="text-right">
          <span className="text-3xl font-black text-orange-500">{riskScore}</span>
          <span className="text-slate-500 text-sm ml-1">/ 100</span>
        </div>
      </div>

      {/* 리스크 점수 게이지 시각화 (간이 구현) */}
      <div className="relative h-4 bg-slate-700 rounded-full mb-8 overflow-hidden">
        <motion.div 
          initial={{ width: 0 }}
          animate={{ width: `${riskScore}%` }}
          transition={{ duration: 1, ease: "easeOut" }}
          className={`h-full ${riskScore > 70 ? 'bg-linear-to-r from-orange-500 to-red-600' : 'bg-blue-500'}`}
        />
      </div>

      {/* "Top Risk 요인" 막대 그래프 영역 */}
      <div className="space-y-5">
        <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-tighter">Top Risk Factors Analysis</h4>
        {riskFactors.map((factor, index) => (
          <div key={index} className="space-y-1">
            <div className="flex justify-between text-xs">
              <span className="text-slate-300">{factor.label}</span>
              <span className="text-slate-400 font-mono">{factor.value}%</span>
            </div>
            <div className="h-1.5 bg-slate-900 rounded-full overflow-hidden">
              <motion.div 
                initial={{ width: 0 }}
                animate={{ width: `${factor.value}%` }}
                transition={{ delay: 0.2 + index * 0.1 }}
                className={`h-full ${factor.color}`}
              />
            </div>
          </div>
        ))}
      </div>

    </div>
  );
}