"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Info, ChevronDown, ChevronUp } from "lucide-react";

export default function Row2RiskDetail() {
  const [isExpanded, setIsExpanded] = useState(false);
  const riskScore = 72;
  
  const riskFactors = [
    { label: "예상 초과 확률", value: 85, color: "bg-red-500" },
    { label: "기준 대비 여유도(Margin)", value: 40, color: "bg-orange-500" },
    { label: "예상 변화량", value: 65, color: "bg-yellow-500" },
    { label: "데이터 신뢰도", value: 95, color: "bg-emerald-500" },
    { label: "외생요인(강우·계절)", value: 30, color: "bg-blue-500" },
  ];

  return (
    <div 
      onClick={() => setIsExpanded(!isExpanded)}
      className="bg-slate-800/40 p-5 rounded-3xl border border-white/10 shadow-xl cursor-pointer hover:bg-slate-800/60 transition-all select-none"
    >
      <div className="flex justify-between items-center mb-4">
        <div>
          <h3 className="text-md font-bold text-white flex items-center gap-2">
            운영 리스크 점수 상세 <Info className="w-3.5 h-3.5 text-slate-500" />
          </h3>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-2xl font-black text-orange-500">{riskScore}<span className="text-xs text-slate-500 font-normal ml-0.5">/ 100</span></span>
          <div className="p-1 rounded-full bg-white/5">
            {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
          </div>
        </div>
      </div>

      {/* 메인 프로그레스 바 */}
      <div className="relative h-3.5 bg-slate-900/50 rounded-full overflow-hidden border border-white/5">
        <motion.div 
          initial={{ width: 0 }}
          animate={{ width: `${riskScore}%` }}
          transition={{ duration: 1 }}
          className="h-full bg-linear-to-r from-orange-600 to-red-600"
        />
      </div>

      <AnimatePresence>
        {isExpanded && (
          <motion.div 
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden"
          >
            <div className="pt-6 space-y-4 mt-4 border-t border-white/5">
              {riskFactors.map((factor, i) => (
                <div key={i} className="space-y-1.5">
                  <div className="flex justify-between text-[11px] font-medium">
                    <span className="text-slate-400">{factor.label}</span>
                    <span className="text-slate-200">{factor.value}%</span>
                  </div>
                  <div className="h-1 bg-slate-900 rounded-full overflow-hidden">
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
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}