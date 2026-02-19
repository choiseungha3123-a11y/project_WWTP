import DashboardClient from "./DashboardClient";

// ISR 설정 (서버 전용)
export const revalidate = 60 * 30; 

export default function DashboardPage() {
  return <DashboardClient />;
}