"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import AddMemberModal from "../../options/AddMemberModal";

interface Member {
  userNo: number;
  userId: string;
  userName: string;
  userEmail: string; // 백엔드 필드명 매칭
  role: string;
  validateEmail: boolean; // 인증 여부 확인용
}

export default function MemberManagementPage() {
  const router = useRouter();
  const [members, setMembers] = useState<Member[]>([]);
  const [isAddModalOpen, setIsAddModalOpen] = useState(false);
  const [loading, setLoading] = useState(true);

  // 사원 리스트 불러오기
  const fetchMembers = async () => {
    setLoading(true);

    try {
      const response = await fetch("/api/member/list", { 
        headers: { 
          "Authorization": `Bearer ${localStorage.getItem('accessToken')}`
        }
      });
      const result = await response.json();
      
      if (result.success) {
        setMembers(result.dataList || []);
      } else {
        alert(result.errorMsg);
        if (result.errorMsg.includes("토큰") || result.errorMsg.includes("로그인")) {
          router.replace("/");
        }
      }
    } catch (error) {
      console.error("멤버 로드 오류:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMembers();
  }, []);

  // 이메일 인증 발송 핸들러
  const handleSendAuthEmail = async (userNo: number, userEmail: string) => {
    if (!confirm(`${userEmail} 주소로 인증 메일을 발송하시겠습니까?`)) return;

    try {
      const res = await fetch("/api/member/validateEmail", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${localStorage.getItem('accessToken')}`
        },
        // 백엔드 validEmailDTO가 { userNo: long } 형태이므로 이에 맞춤
        body: JSON.stringify({ userNo: userNo })
      });

      const result = await res.json();

      if (result.success) {
        alert("인증 메일이 성공적으로 발송되었습니다.");
      } else {
        alert(result.errorMsg || "메일 발송에 실패했습니다.");
      }
    } catch (error) {
      console.error("메일 발송 오류:", error);
      alert("서버 통신 중 오류가 발생했습니다.");
    }
  };

  const handleResetPassword = async (userNo: number, userId: string) => {
    const newPassword = `${userId}1234`;

    if (!confirm(`${userId}님의 비밀번호를 '${newPassword}'로 초기화하시겠습니까?`)) return;

    try {
      const res = await fetch("/api/member/modify", {
        method: "PATCH", 
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${localStorage.getItem('accessToken')}`
        },
        body: JSON.stringify({
          userNo: userNo,
          password: newPassword
        })
      });

      const result = await res.json();
      if (result.success) {
        alert(`비밀번호가 성공적으로 초기화되었습니다.\n새 비밀번호: ${newPassword}`);
      } else {
        alert(result.errorMsg || "초기화에 실패했습니다.");
      }
    } catch (error) {
      console.error("비밀번호 초기화 오류:", error);
      alert("서버와 통신 중 오류가 발생했습니다.");
    }
  };


  const handleDelete = async (userNo: number, userId: string) => {
    if (!confirm(`${userId}(${userNo}) 사원을 삭제하시겠습니까?`)) return;
    
    try {
      const res = await fetch("/api/member/delete", {
        method: "DELETE",
        headers: { 
          "Content-Type": "application/json",
          "Authorization": `Bearer ${localStorage.getItem('accessToken')}` 
        },
        body: JSON.stringify({ userNo, userId })
      });
      const result = await res.json();
      if (result.success) {
        alert("삭제 완료");
        fetchMembers();
      }
    } catch (error) {
      console.error("삭제 오류:", error);
      alert("삭제 실패");
    }
  };

  return (
    <div className="min-h-screen bg-slate-900 text-white p-8 font-sans">
      {/* 상단 네비게이션 */}
      <div className="max-w-6xl mx-auto flex justify-between items-end mb-10">
        <div>
          <button 
            onClick={() => router.push("/dashboard")}
            className="text-slate-300 hover:text-blue-400 text-medium mb-4 transition-colors flex items-center gap-1"
          >
            🏠 대시보드로 돌아가기
          </button>
          <h1 className="text-4xl font-black tracking-tight text-white">
            사원 <span className="text-blue-500">관리</span>
          </h1>
        </div>
        
        <button 
          onClick={() => setIsAddModalOpen(true)}
          className="bg-blue-600 hover:bg-blue-500 text-white px-6 py-3 rounded-2xl font-bold transition-all shadow-lg shadow-blue-500/20 active:scale-95"
        >
          + 신규 사원 등록
        </button>
      </div>

      {/* 리스트 테이블 */}
<div className="max-w-6xl mx-auto bg-slate-800/40 rounded-3xl border border-white/10 overflow-hidden backdrop-blur-md shadow-2xl">
  <table className="w-full text-sm sm:text-base table-fixed"> 
    <thead>
      <tr className="border-b border-white/5 bg-white/5 text-slate-400 text-xs uppercase tracking-wider">
        <th className="p-6 text-center font-semibold w-16">No</th>
        <th className="p-6 text-center font-semibold">아이디</th>
        <th className="p-6 text-center font-semibold w-1/4">이메일</th> 
        <th className="p-6 text-center font-semibold">이름</th>
        <th className="p-6 text-center font-semibold">권한</th>
        <th className="p-6 text-center font-semibold">관리 액션</th>
      </tr>
    </thead>
    <tbody className="divide-y divide-white/5">
      {loading ? (
        <tr><td colSpan={6} className="p-20 text-center text-slate-500 animate-pulse">데이터 로딩 중...</td></tr>
      ) : members.length === 0 ? (
        <tr><td colSpan={6} className="p-20 text-center text-slate-500">등록된 사원이 없습니다.</td></tr>
      ) : (
        members.map((mem) => (
          <motion.tr 
            initial={{ opacity: 0 }} 
            animate={{ opacity: 1 }} 
            key={mem.userNo} 
            className="hover:bg-white/5 transition-colors group"
          >
            <td className="p-6 text-center text-slate-500 text-sm">{mem.userNo}</td>
            <td className="p-6 text-center font-bold text-blue-100">{mem.userId}</td>
            
            {/* 이메일 셀 */}
            <td className="p-6 text-center">
              <div className="flex flex-col items-center justify-center gap-1.5">
                <div 
                  className="max-w-xs text-slate-400 text-sm italic truncate" 
                  title={mem.userEmail} 
                >
                  {mem.userEmail || "-"}
                </div>
                {/* 이메일이 있고 인증되지 않았을 때만 버튼 표시 */}
                {mem.userEmail && (
                  mem.validateEmail ? (
                    <span className="text-[10px] bg-green-500/10 text-green-400 px-2 py-0.5 rounded border border-green-500/20">
                      인증 완료됨
                    </span>
                  ) : (
                    <button
                      onClick={() => handleSendAuthEmail(mem.userNo, mem.userEmail)}
                      className="text-[10px] bg-blue-500/10 hover:bg-blue-500/30 text-blue-400 px-2 py-0.5 rounded border border-blue-500/20 transition-all"
                    >
                      📩 인증 메일 발송
                    </button>
                  )
                )}
              </div>
            </td>
            
            <td className="p-6 text-center text-slate-300 font-medium">{mem.userName}</td>
            <td className="p-6 text-center text-sm">
              <span className={`inline-block px-3 py-1 rounded-full text-xs ${mem.role === 'ROLE_ADMIN' ? 'bg-purple-500/10 text-purple-400 border border-purple-500/20' : 'bg-blue-500/10 text-blue-400 border border-blue-500/20'}`}>
                {mem.role}
              </span>
            </td>
            <td className="p-6 text-center">
              <div className="flex justify-center gap-2">
                <button 
                  onClick={() => handleResetPassword(mem.userNo, mem.userId)}
                  className="bg-slate-700 hover:bg-slate-600 text-slate-200 px-3 py-1.5 rounded-lg text-xs transition-colors whitespace-nowrap"
                >
                  비밀번호 초기화
                </button>
                <button 
                  onClick={() => handleDelete(mem.userNo, mem.userId)}
                  className="bg-red-500/10 hover:bg-red-500 text-red-500 hover:text-white px-3 py-1.5 rounded-lg text-xs transition-all border border-red-500/20"
                >
                  삭제
                </button>
              </div>
            </td>
          </motion.tr>
        ))
      )}
    </tbody>
  </table>
</div>

      <AddMemberModal 
        isOpen={isAddModalOpen} 
        onClose={() => setIsAddModalOpen(false)} 
        onSuccess={fetchMembers} 
      />
    </div>
  );
}