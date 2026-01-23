// src/services/authService.ts

const BASE_URL = 'http://localhost:8080/api';

export const authService = {
  // email과 password의 타입을 string으로 지정하여 오류를 방지합니다.
  login: async (email: string, password: string) => {
    const response = await fetch(`${BASE_URL}/login`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ email, password }),
    });
    
    // 서버에서 에러 응답(400, 401, 500 등)이 왔을 때 처리
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.message || '로그인에 실패했습니다.');
    }
    
    return response.json(); // 성공 시 서버에서 보낸 데이터(토큰 등) 반환
  },
};