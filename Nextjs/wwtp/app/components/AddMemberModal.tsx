"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";

interface AddMemberModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: () => void;
}

export default function AddMemberModal({ isOpen, onClose, onSuccess }: AddMemberModalProps) {
  const [formData, setFormData] = useState({
    userId: "",
    userName: "",
    password: "",
    confirmPassword: "",
    role: "ROLE_MEMBER",
  });
  const [loading, setLoading] = useState(false);

  // 모달이 열릴 때마다 입력창 초기화
  useEffect(() => {
    if (isOpen) {
      setFormData({
        userId: "",
        userName: "",
        password: "",
        confirmPassword: "",
        role: "ROLE_MEMBER",
      });
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (formData.password !== formData.confirmPassword) {
      alert("비밀번호가 일치하지 않습니다.");
      return;
    }

    setLoading(true);
    try {
      const response = await fetch("/api/member/saveMember", { // 백엔드 엔드포인트 확인 필요
        method: "POST",
        headers: { 
          "Content-Type": "application/json",
          "Authorization": `Bearer ${localStorage.getItem('accessToken')}` 
        },
        body: JSON.stringify({
          userId: formData.userId,
          userName: formData.userName,
          password: formData.password,
          role: formData.role
        }),
      });

      const result = await response.json();

      if (result.success) {
        alert("새로운 회원이 성공적으로 등록되었습니다.");
        onSuccess();
        onClose();
      } else {
        alert(result.errorMsg || "등록에 실패했습니다.");
      }
    } catch (error) {
      alert("서버 통신 오류가 발생했습니다.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-[110] flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
      <motion.div 
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="bg-white text-slate-800 w-full max-w-md rounded-2xl shadow-2xl overflow-hidden"
      >
        <div className="p-6 border-b flex justify-between items-center bg-slate-50">
          <h2 className="text-xl font-bold text-blue-600">신규 회원 등록</h2>
          <button onClick={onClose} className="text-slate-400 hover:text-slate-600">✕</button>
        </div>

        <form onSubmit={handleSubmit} className="p-6 space-y-4">
          <div>
            <label className="block text-xs font-semibold text-slate-500 mb-1">아이디 (ID)</label>
            <input 
              type="text" 
              required
              value={formData.userId}
              onChange={(e) => setFormData({...formData, userId: e.target.value})}
              className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 outline-none"
              placeholder="접속 아이디 입력"
            />
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-500 mb-1">사용자 이름</label>
            <input 
              type="text" 
              required
              value={formData.userName}
              onChange={(e) => setFormData({...formData, userName: e.target.value})}
              className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 outline-none"
              placeholder="실명 입력"
            />
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs font-semibold text-slate-500 mb-1">비밀번호</label>
              <input 
                type="password" 
                required
                value={formData.password}
                onChange={(e) => setFormData({...formData, password: e.target.value})}
                className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 outline-none"
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-500 mb-1">비밀번호 확인</label>
              <input 
                type="password" 
                required
                value={formData.confirmPassword}
                onChange={(e) => setFormData({...formData, confirmPassword: e.target.value})}
                className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 outline-none"
              />
            </div>
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-500 mb-1">권한 설정</label>
            <select 
              value={formData.role}
              onChange={(e) => setFormData({...formData, role: e.target.value})}
              className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 outline-none bg-white"
            >
              <option value="ROLE_MEMBER">일반 사원 (Member)</option>
              <option value="ROLE_ADMIN">관리자 (Admin)</option>
            </select>
          </div>

          <div className="flex gap-3 mt-6">
            <button type="button" onClick={onClose} className="flex-1 py-3 bg-slate-100 rounded-xl font-medium">취소</button>
            <button 
              type="submit" 
              disabled={loading}
              className="flex-1 py-3 bg-blue-600 text-white rounded-xl font-bold hover:bg-blue-700 disabled:bg-blue-300"
            >
              {loading ? "등록 중..." : "회원 추가"}
            </button>
          </div>
        </form>
      </motion.div>
    </div>
  );
}