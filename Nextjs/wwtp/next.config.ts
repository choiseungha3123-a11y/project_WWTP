import type { NextConfig } from "next";
import { loadEnvConfig } from '@next/env';

const projectDir = process.cwd();
loadEnvConfig(projectDir);

const nextConfig: NextConfig = {
  async rewrites() {
    const backendUrl = process.env.NEXT_PUBLIC_API_URL;
    const destinationUrl = backendUrl || "http://10.125.121.176:8081";

    return [
      {
        source: '/api/:path*',
        destination: `${destinationUrl}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;