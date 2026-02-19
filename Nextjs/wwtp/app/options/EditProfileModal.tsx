"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";

interface EditProfileModalProps {
  isOpen: boolean;
  onClose: () => void;
  currentUser: {
    userNo: number;
    id: string;
    name: string;
    role: string;
    email?: string;
  };
  onUpdateSuccess: (newId: string, newName: string, newEmail: string) => void;
}

export default function EditProfileModal({ 
  isOpen, 
  onClose, 
  currentUser, 
  onUpdateSuccess 
}: EditProfileModalProps) {
  const [userId, setUserId] = useState<string>(currentUser.id);
  const [username, setUsername] = useState<string>(currentUser.name);
  const [userEmail, setUserEmail] = useState<string>(currentUser.email || ""); // 이메일 상태 추가
  const [password, setPassword] = useState<string>("");
  const [confirmPassword, setConfirmPassword] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);

  // 모달이 열릴 때 초기 데이터 세팅
  useEffect(() => {
    if (isOpen) {
      setUserId(currentUser.id);
      setUsername(currentUser.name);
      setUserEmail(currentUser.email || "");
      setPassword(""); 
      setConfirmPassword("");
    }
  }, [isOpen, currentUser]);

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    // 비밀번호 일치 확인
    if (password && password !== confirmPassword) {
      alert("비밀번호가 일치하지 않습니다.");
      return;
    }

    setLoading(true);
    try {
      // 백엔드 memberModifyDTO 구조에 맞춰 데이터 전송
      const response = await fetch(`/api/member/modify`, {
        method: "PATCH",
        headers: { 
          "Content-Type": "application/json", 
          "Authorization": `${localStorage.getItem("accessToken")}` 
        },
        body: JSON.stringify({
          userNo: currentUser.userNo,
          userId: userId,
          password: password, 
          userName: username,
          userEmail: userEmail, // 이메일 추가
          role: currentUser.role
        }),
      });

      const result = await response.json();

      if (result.success) {
        alert("정보가 성공적으로 수정되었습니다.");
        
        // 로컬스토리지 업데이트
        localStorage.setItem("userId", userId);
        localStorage.setItem("userName", username);
        localStorage.setItem("userEmail", userEmail);
        
        // 부모 컴포넌트(Dashboard) 상태 업데이트 (인자 3개 전달)
        onUpdateSuccess(userId, username, userEmail);
        onClose();
      } else {
        alert(result.errorMsg || "수정에 실패했습니다.");
      }
    } catch (error) {
      console.error("Error:", error);
      alert("서버 통신 오류가 발생했습니다.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-100 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
      <motion.div 
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="bg-white text-slate-800 w-full max-w-md rounded-2xl shadow-2xl overflow-hidden"
      >
        <div className="p-6 border-b flex justify-between items-center">
          <h2 className="text-xl font-bold text-slate-900">개인정보 수정</h2>
          <button onClick={onClose} className="text-slate-400 hover:text-slate-600 transition-colors">✕</button>
        </div>

        <form onSubmit={handleSubmit} className="p-6 space-y-4">
          {/* 아이디 입력 */}
          <div>
            <label className="block text-xs font-semibold text-slate-500 mb-1">아이디 (ID)</label>
            <input 
              type="text" 
              value={userId}
              onChange={(e) => setUserId(e.target.value)}
              className="w-full px-4 py-2 border border-slate-200 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none transition-all text-slate-700"
              required
            />
          </div>

          {/* 이름 입력 */}
          <div>
            <label className="block text-xs font-semibold text-slate-500 mb-1">사용자 이름</label>
            <input 
              type="text" 
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="w-full px-4 py-2 border border-slate-200 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none transition-all text-slate-700"
              required
            />
          </div>

          {/* 이메일 입력 추가 */}
          <div>
            <label className="block text-xs font-semibold text-slate-500 mb-1">이메일 주소</label>
            <input 
              type="email" 
              placeholder="example@email.com"
              value={userEmail}
              onChange={(e) => setUserEmail(e.target.value)}
              className="w-full px-4 py-2 border border-slate-200 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none transition-all text-slate-700"
              required
            />
          </div>

          <hr className="border-slate-100 my-2" />

          {/* 비밀번호 입력 */}
          <div>
            <label className="block text-xs font-semibold text-slate-500 mb-1">새 비밀번호</label>
            <input 
              type="password" 
              placeholder="10~20자, 대소문자/숫자/특수문자 포함"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full px-4 py-2 border border-slate-200 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none transition-all text-slate-700"
              required
            />
          </div>

          {/* 비밀번호 확인 입력 */}
          <div>
            <label className="block text-xs font-semibold text-slate-500 mb-1">비밀번호 확인</label>
            <input 
              type="password" 
              placeholder="동일한 비밀번호 입력"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              className="w-full px-4 py-2 border border-slate-200 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none transition-all text-slate-700"
              required
            />
          </div>

          <div className="flex gap-3 mt-6">
            <button 
              type="button" 
              onClick={onClose} 
              className="flex-1 py-3 bg-slate-100 text-slate-600 rounded-xl font-medium hover:bg-slate-200 transition-colors"
            >
              취소
            </button>
            <button 
              type="submit" 
              disabled={loading}
              className="flex-1 py-3 bg-blue-600 text-white rounded-xl font-medium hover:bg-blue-700 disabled:bg-blue-300 transition-colors shadow-lg shadow-blue-500/20"
            >
              {loading ? "저장 중..." : "수정 완료"}
            </button>
          </div>
        </form>
      </motion.div>
    </div>
  );
}