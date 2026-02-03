"use client";

import { useState } from "react";

export default function Row5ActionPanel() {
  const [memo, setMemo] = useState("");

  const handleAction = async (type: 'apply' | 'hold' | 'ignore') => {
    // API 연동 로직 (운영 이력 DB 저장)
    console.log(`Action: ${type}, Memo: ${memo}`);
    alert(`조치가 [${type}] 상태로 기록되었습니다.`);
  };

  return (
    <div className="bg-slate-800/50 p-6 rounded-3xl border border-white/10 h-full">
      <h3 className="text-lg font-bold mb-4 text-blue-400">Row5 운영 권고 & 조치 이력</h3>
      
      {/* 자동 권고 영역 */}
      <div className="space-y-2 mb-6">
        <div className="bg-blue-500/10 p-3 rounded-xl border border-blue-500/20 text-sm">
          💡 송풍량 +10% 권장 (질산화 저하 위험)
        </div>
        <div className="bg-blue-500/10 p-3 rounded-xl border border-blue-500/20 text-sm">
          💡 응집제 투입량 조정 제안
        </div>
      </div>

      {/* 관리자 액션 & 메모 */}
      <div className="space-y-4 pt-4 border-t border-white/5">
        <textarea 
          value={memo}
          onChange={(e) => setMemo(e.target.value)}
          placeholder="조치 사항을 입력하세요 (메모)"
          className="w-full bg-slate-900 border border-white/10 rounded-xl p-3 text-sm focus:outline-none focus:border-blue-500 transition-all h-24"
        />
        
        <div className="flex gap-2">
          <button onClick={() => handleAction('apply')} className="flex-1 bg-blue-600 hover:bg-blue-500 py-2 rounded-lg font-bold transition-all text-sm">적용함</button>
          <button onClick={() => handleAction('hold')} className="flex-1 bg-slate-700 hover:bg-slate-600 py-2 rounded-lg font-bold transition-all text-sm">보류</button>
          <button onClick={() => handleAction('ignore')} className="flex-1 bg-red-500/20 hover:bg-red-500/40 text-red-400 py-2 rounded-lg font-bold transition-all text-sm">무시</button>
        </div>
      </div>

      <div className="mt-4 text-[11px] text-slate-500 text-right italic">
        * 시간·담당자 자동 기록 → 운영 이력 DB
      </div>
    </div>
  );
}