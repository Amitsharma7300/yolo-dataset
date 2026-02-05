/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  images: {
    domains: ['localhost'],
  },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://35.240.168.55:8080/api/:path*',
      },
    ];
  },
}

module.exports = nextConfig
