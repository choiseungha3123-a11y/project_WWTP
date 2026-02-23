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
    email: "",
    password: "",
    confirmPassword: "",
    role: "ROLE_MEMBER",
  });
  
  const [loading, setLoading] = useState(false);
  const [emailError, setEmailError] = useState(""); // 이메일 중복/형식 에러 메시지

  // 아이디 입력 시 자동으로 초기 비밀번호 세팅 (사용자 편의 기능)
  useEffect(() => {
    if (formData.userId) {
      const defaultPw = `${formData.userId}1234`;
      setFormData(prev => ({
        ...prev,
        password: defaultPw,
        confirmPassword: defaultPw
      }));
    } else {
      setFormData(prev => ({
        ...prev,
        password: "",
        confirmPassword: ""
      }));
    }
  }, [formData.userId]);

  // 모달이 열릴 때 상태 초기화
  useEffect(() => {
    if (isOpen) {
      setFormData({
        userId: "",
        userName: "",
        email: "",
        password: "",
        confirmPassword: "",
        role: "ROLE_MEMBER",
      });
      setEmailError("");
    }
  }, [isOpen]);

  // 백엔드 API를 이용한 이메일 중복 체크 (선택 사항)
  const checkEmailDuplicate = async (email: string) => {
    if (!email || !email.includes('@')) return;
    try {
      const res = await fetch(`/api/member/checkEmail?userEmail=${email}`);
      const result = await res.json();
      if (!result.success) {
        setEmailError(result.errorMsg || "이미 사용 중인 이메일입니다.");
      } else {
        setEmailError("");
      }
    } catch (err) {
      console.error("이메일 중복 체크 실패", err);
    }
  };

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (formData.password !== formData.confirmPassword) {
      alert("비밀번호가 일치하지 않습니다.");
      return;
    }

    if (emailError) {
      alert("이메일을 확인해주세요.");
      return;
    }

    setLoading(true);
    try {
      const response = await fetch("/api/member/create", { 
        method: "PUT",
        headers: { 
          "Content-Type": "application/json",
          "Authorization": `Bearer ${localStorage.getItem('accessToken')}` 
        },
        body: JSON.stringify({
          userId: formData.userId,
          password: formData.password,
          userName: formData.userName,
          userEmail: formData.email, // 백엔드 DTO 필드명인 'userEmail'에 매칭
        }),
      });

      const result = await response.json();

      if (result.success) {
        alert(`새로운 회원이 성공적으로 등록되었습니다.\n초기 비밀번호는 [ ${formData.password} ] 입니다.`);
        onSuccess();
        onClose();
      } else {
        alert(result.errorMsg || "등록에 실패했습니다.");
      }
    } catch (error) {
      console.error("회원 등록 에러:", error);
      alert("서버 통신 오류가 발생했습니다.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-110 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
      <motion.div 
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="bg-white text-slate-800 w-full max-w-md rounded-2xl shadow-2xl overflow-hidden"
      >
        <div className="p-6 border-b flex justify-between items-center bg-slate-50">
          <h2 className="text-xl font-bold text-blue-600">신규 회원 등록</h2>
          <button onClick={onClose} className="text-slate-400 hover:text-slate-600 transition-colors">✕</button>
        </div>

        <form onSubmit={handleSubmit} className="p-6 space-y-4">
          {/* 아이디 영역 */}
          <div>
            <label className="block text-xs font-semibold text-slate-500 mb-1">아이디 (ID)</label>
            <input 
              type="text" 
              required
              value={formData.userId}
              onChange={(e) => setFormData({...formData, userId: e.target.value})}
              className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 outline-none transition-all"
              placeholder="접속 아이디 입력"
            />
          </div>

          {/* 이름 영역 */}
          <div>
            <label className="block text-xs font-semibold text-slate-500 mb-1">사용자 이름</label>
            <input 
              type="text" 
              required
              value={formData.userName}
              onChange={(e) => setFormData({...formData, userName: e.target.value})}
              className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 outline-none transition-all"
              placeholder="실명 입력"
            />
          </div>

          {/* 이메일 영역 (추가됨) */}
          <div>
            <label className="block text-xs font-semibold text-slate-500 mb-1">사용자 이메일</label>
            <input 
              type="email" 
              required
              value={formData.email}
              onChange={(e) => setFormData({...formData, email: e.target.value})}
              onBlur={() => checkEmailDuplicate(formData.email)} // 입력 후 중복 체크 수행
              className={`w-full px-4 py-2 border rounded-lg focus:ring-2 outline-none transition-all ${emailError ? 'border-red-500 focus:ring-red-200' : 'focus:ring-blue-500'}`}
              placeholder="example@email.com"
            />
            {emailError && <p className="text-[10px] text-red-500 mt-1 ml-1">{emailError}</p>}
          </div>

          {/* 비밀번호 영역 */}
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs font-semibold text-slate-500 mb-1">비밀번호</label>
              <input 
                type="password" 
                required
                value={formData.password}
                onChange={(e) => setFormData({...formData, password: e.target.value})}
                className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 outline-none transition-all"
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-500 mb-1">비밀번호 확인</label>
              <input 
                type="password" 
                required
                value={formData.confirmPassword}
                onChange={(e) => setFormData({...formData, confirmPassword: e.target.value})}
                className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 outline-none transition-all"
              />
            </div>
          </div>

          {/* 권한 설정 영역 */}
          <div>
            <label className="block text-xs font-semibold text-slate-500 mb-1">권한 설정</label>
            <select 
              value={formData.role}
              onChange={(e) => setFormData({...formData, role: e.target.value})}
              className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 outline-none bg-white transition-all cursor-pointer"
            >
              <option value="ROLE_MEMBER">일반 사원 (Member)</option>
              <option value="ROLE_VIEWER">뷰어 (Viewer)</option>
              <option value="ROLE_ADMIN">관리자 (Admin)</option>
            </select>
          </div>

          {/* 하단 버튼 영역 */}
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
              disabled={loading || !!emailError}
              className="flex-1 py-3 bg-blue-600 text-white rounded-xl font-bold hover:bg-blue-700 disabled:bg-blue-300 transition-all shadow-lg shadow-blue-200"
            >
              {loading ? "등록 중..." : "회원 추가"}
            </button>
          </div>
        </form>
      </motion.div>
    </div>
  );
}