"use client";

export default function Row3Charts() {
  return (
    <div className="grid grid-cols-2 gap-4">
      {/* 유입유량 & 수위 예측 */}
      <div className="bg-slate-800/40 p-5 rounded-3xl border border-white/10 min-h-55 flex flex-col">
        <h3 className="text-xs font-bold text-slate-400 mb-4">유입유량 & 수위 그래프 (예측)</h3>
        <div className="flex-1 bg-slate-900/60 rounded-xl border border-dashed border-white/10 flex items-center justify-center relative overflow-hidden">
          {/* 그래프 자리 표시용 더미 (실제 구현 시 Recharts 사용) */}
          <div className="absolute inset-0 flex items-center justify-center opacity-20">
             <div className="w-full h-px bg-red-500 absolute top-1/2" /> {/* 기준선 */}
             <p className="text-[10px]">Graph Area (Flow/Level)</p>
          </div>
          <span className="text-[10px] absolute bottom-2 right-2 text-blue-400 font-mono italic">Future Prediction →</span>
        </div>
      </div>

      {/* 수질 항목 예측 */}
      <div className="bg-slate-800/40 p-5 rounded-3xl border border-white/10 min-h-55 flex flex-col">
        <h3 className="text-xs font-bold text-slate-400 mb-4">TOC / TN / TP / SS 예측</h3>
        <div className="flex-1 bg-slate-900/60 rounded-xl border border-dashed border-white/10 flex items-center justify-center relative overflow-hidden">
          <div className="absolute inset-0 flex items-center justify-center opacity-20">
             <div className="w-full h-px bg-green-500 absolute top-1/3" />
             <p className="text-[10px]">Graph Area (Water Quality)</p>
          </div>
          <div className="absolute top-2 right-2 flex gap-2">
            <span className="text-[9px] text-emerald-400">● TOC</span>
            <span className="text-[9px] text-blue-400">● TN</span>
          </div>
        </div>
      </div>
    </div>
  );
}