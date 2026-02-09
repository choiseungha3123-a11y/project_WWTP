"use client";

import { useState } from "react";
import { Send, Clock, AlertCircle } from "lucide-react";

export default function Row5ActionPanel() {
  const [memo, setMemo] = useState("");

  const handleAction = async (type: 'apply' | 'hold' | 'ignore') => {
    console.log(`Action: ${type}, Memo: ${memo}`);
    alert(`조치가 [${type}] 상태로 기록되었습니다.`);
  };

  return (
    // p-6 -> p-5로 줄여 내부 공간 확보
    <div className="bg-slate-800/50 p-5 rounded-3xl border border-white/10 h-full flex flex-col shadow-lg">
      <div className="flex justify-between items-center mb-3 shrink-0">
        <h3 className="text-[15px] font-bold text-blue-400 flex items-center gap-2">
          <AlertCircle className="w-4 h-4" /> 운영 권고 & 조치
        </h3>
        <span className="text-[10px] text-slate-500 flex items-center gap-1">
          <Clock className="w-3 h-3" /> 자동 기록 중
        </span>
      </div>
      
      {/* 자동 권고 영역: flex-none으로 고정하거나 필요시 스크롤 */}
      <div className="space-y-2 mb-3 shrink-0">
        <div className="bg-blue-500/10 px-3 py-2.5 rounded-xl border border-blue-500/20 text-xs font-medium text-blue-100 flex items-center gap-2">
          <span className="text-blue-400">💡</span> 송풍량 +10% 권장 (질산화 저하 위험)
        </div>
        <div className="bg-indigo-500/10 px-3 py-2.5 rounded-xl border border-indigo-500/20 text-xs font-medium text-indigo-100 flex items-center gap-2">
          <span className="text-indigo-400">⚡</span> 응집제 투입량 조정 제안
        </div>
      </div>

      {/* 구분선 */}
      <div className="border-t border-white/5 my-1 shrink-0"></div>

      {/* 관리자 액션 & 메모: 남는 공간(flex-1)을 모두 메모창이 차지하도록 설정 */}
      <div className="flex-1 flex flex-col gap-3 pt-2 min-h-0">
        <textarea 
          value={memo}
          onChange={(e) => setMemo(e.target.value)}
          placeholder="조치 사항을 입력하세요 (메모)"
          // h-24 제거 -> flex-1로 변경하여 남는 높이 자동 채움
          className="flex-1 w-full bg-slate-900/50 border border-white/10 rounded-xl p-3 text-xs text-slate-200 focus:outline-none focus:border-blue-500 transition-all resize-none placeholder:text-slate-600"
        />
        
        {/* 버튼 영역: 높이 고정 (shrink-0) */}
        <div className="flex gap-2 shrink-0">
          <button 
            onClick={() => handleAction('apply')} 
            className="flex-1 bg-blue-600 hover:bg-blue-500 py-2.5 rounded-lg font-bold transition-all text-xs text-white shadow-lg shadow-blue-900/20 flex justify-center items-center gap-1"
          >
            <Send className="w-3 h-3" /> 적용
          </button>
          <button 
            onClick={() => handleAction('hold')} 
            className="flex-1 bg-slate-700 hover:bg-slate-600 py-2.5 rounded-lg font-bold transition-all text-xs text-slate-200"
          >
            보류
          </button>
          <button 
            onClick={() => handleAction('ignore')} 
            className="flex-1 bg-red-500/10 hover:bg-red-500/20 text-red-400 border border-red-500/20 py-2.5 rounded-lg font-bold transition-all text-xs"
          >
            무시
          </button>
        </div>
      </div>
    </div>
  );
}