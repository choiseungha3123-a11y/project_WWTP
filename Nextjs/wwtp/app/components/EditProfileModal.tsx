"use client";

import { useState } from "react";
import { motion } from "framer-motion";

interface EditProfileModalProps {
  isOpen: boolean;
  onClose: () => void;
  currentUser: {
    id : string;
    name: string;
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

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (password && password !== confirmPassword) {
      alert("비밀번호가 일치하지 않습니다.");
      return;
    }

    setLoading(true);
    try {
      const response = await fetch("/api/member/modifyMember", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          username,
          password: password || null, 
        }),
      });

      if (response.ok) {
        alert("정보가 성공적으로 수정되었습니다.");
        localStorage.setItem("userId", userId);
        localStorage.setItem("userName", username);
        
        onUpdateSuccess(userId, username);
        onClose();
      } else {
        const errorData = await response.json();
        alert(errorData.message || "수정에 실패했습니다. 아이디 중복 여부를 확인하세요.");
      }
    } catch (error) {
      console.error("Error:", error);
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
              className="w-full px-4 py-2 border rounded-lg outline-none focus:ring-2 focus:ring-blue-500 font-medium"
              required
            />
            <p className="text-[10px] text-slate-400 mt-1">* 로그인 시 사용하는 아이디가 변경됩니다.</p>
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-500 mb-1">사용자 이름</label>
            <input 
              type="text" 
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="w-full px-4 py-2 border rounded-lg outline-none focus:ring-2 focus:ring-blue-500"
              required
            />
          </div>

          <hr className="border-slate-100" />

          <div>
            <label className="block text-xs font-semibold text-slate-500 mb-1">새 비밀번호</label>
            <input 
              type="password" 
              placeholder="변경 시에만 입력"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full px-4 py-2 border rounded-lg outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-500 mb-1">비밀번호 확인</label>
            <input 
              type="password" 
              placeholder="비밀번호 확인"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              className="w-full px-4 py-2 border rounded-lg outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div className="flex gap-3 mt-6">
            <button type="button" onClick={onClose} className="flex-1 py-3 bg-slate-100 rounded-xl font-medium">취소</button>
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