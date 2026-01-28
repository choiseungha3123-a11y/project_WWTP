"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";

interface EditProfileModalProps {
  isOpen: boolean;
  onClose: () => void;
  currentUser: {
    userNo: number;
    id : string;
    name: string;
    role: string;
  };
  onUpdateSuccess: (newId: string, newName: string) => void;
}

export default function EditProfileModal({ 
  isOpen, 
  onClose, 
  currentUser, 
  onUpdateSuccess 
}: EditProfileModalProps) {
  const [userId, setUserId] = useState<string>(currentUser.id);
  const [username, setUsername] = useState<string>(currentUser.name);
  const [password, setPassword] = useState<string>("");
  const [confirmPassword, setConfirmPassword] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);

  useEffect(() => {
    if (isOpen) {
      setUserId(currentUser.id);
      setUsername(currentUser.name);
      setPassword(""); 
      setConfirmPassword("");
    }
  }, [isOpen, currentUser]);

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (password && password !== confirmPassword) {
      alert("비밀번호가 일치하지 않습니다.");
      return;
    }

    setLoading(true);
    try {
      console.log(
        'userNo :', currentUser.userNo,
      );
      const response = await fetch(`/api/member/modifyMember`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json", "Authorization": `Bearer ${localStorage.getItem("accessToken")}` },
        body: JSON.stringify({
          userNo: currentUser.userNo,
          userId: userId,
          password: password, 
          userName: username,
          role: currentUser.role
        }),
      });

      const result = await response.json();

      if (result.success) {
        alert("정보가 성공적으로 수정되었습니다.");
        localStorage.setItem("userId", userId);
        localStorage.setItem("userName", username);
        onUpdateSuccess(userId, username);
        onClose();
      } else {
        alert(result.errorMsg);
      }
    } catch (error) {
      console.error("Error:", error);
      alert("서버 통신 오류가 발생했습니다.")
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
          <h2 className="text-xl font-bold">개인정보 수정</h2>
          <button onClick={onClose} className="text-slate-400 hover:text-slate-600">✕</button>
        </div>

        <form onSubmit={handleSubmit} className="p-6 space-y-4">
          <div>
            <label className="block text-xs font-semibold text-slate-500 mb-1">아이디 (ID)</label>
            <input 
              type="text" 
              value={userId}
              onChange={(e) => setUserId(e.target.value)}
              className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 outline-none transition-all"
              required
            />
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-500 mb-1">사용자 이름</label>
            <input 
              type="text" 
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 outline-none transition-all"
              required
            />
          </div>

          <hr className="border-slate-100" />

          <div>
            <label className="block text-xs font-semibold text-slate-500 mb-1">새 비밀번호 (필수)</label>
            <input 
              type="password" 
              placeholder="10~20자, 대소문자/숫자/특수문자 포함"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 outline-none transition-all"
              required
            />
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-500 mb-1">비밀번호 확인</label>
            <input 
              type="password" 
              placeholder="동일한 비밀번호 입력"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 outline-none transition-all"
              required
            />
          </div>

          <div className="flex gap-3 mt-6">
            <button type="button" onClick={onClose} className="flex-1 py-3 bg-slate-100 rounded-xl font-medium transition-colors">취소</button>
            <button 
              type="submit" 
              disabled={loading}
              className="flex-1 py-3 bg-blue-600 text-white rounded-xl font-medium hover:bg-blue-700 disabled:bg-blue-300 transition-colors"
            >
              {loading ? "저장 중..." : "수정 완료"}
            </button>
          </div>
        </form>
      </motion.div>
    </div>
  );
}